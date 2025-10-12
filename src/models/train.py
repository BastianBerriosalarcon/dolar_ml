"""
Script de entrenamiento de modelos con logging y versionado.
Implementa 3 modelos: Random Forest, XGBoost, y ARIMA.
"""
import argparse
import logging
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
try:
    from pmdarima import auto_arima
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
import json

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_metrics(y_true, y_pred):
    """Calcula metricas de evaluacion."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    mape = (np.abs((y_true - y_pred) / y_true).mean()) * 100
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}


def train_random_forest(X_train, y_train, X_val=None, y_val=None):
    """
    Entrena modelo Random Forest.

    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        X_val: Features de validacion (opcional)
        y_val: Target de validacion (opcional)

    Returns:
        Modelo entrenado y metricas si hay datos de validacion
    """
    logger.info("Entrenando Random Forest...")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    metrics = None
    if X_val is not None and y_val is not None:
        y_pred = model.predict(X_val)
        metrics = calculate_metrics(y_val, y_pred)
        logger.info(f"Random Forest - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, R2: {metrics['r2']:.4f}, MAPE: {metrics['mape']:.2f}%")

    return model, metrics


def train_xgboost(X_train, y_train, X_val=None, y_val=None):
    """
    Entrena modelo XGBoost.

    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        X_val: Features de validacion (opcional)
        y_val: Target de validacion (opcional)

    Returns:
        Modelo entrenado y metricas si hay datos de validacion
    """
    logger.info("Entrenando XGBoost...")

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    if X_val is not None and y_val is not None:
        model.set_params(early_stopping_rounds=50)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)

    metrics = None
    if X_val is not None and y_val is not None:
        y_pred = model.predict(X_val)
        metrics = calculate_metrics(y_val, y_pred)
        logger.info(f"XGBoost - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, R2: {metrics['r2']:.4f}, MAPE: {metrics['mape']:.2f}%")

    return model, metrics


def train_arima(y_train, y_val=None):
    """
    Entrena modelo ARIMA usando AutoARIMA.

    Args:
        y_train: Serie temporal de entrenamiento (univariada)
        y_val: Serie temporal de validacion (opcional)

    Returns:
        Modelo entrenado y metricas si hay datos de validacion
    """
    if not ARIMA_AVAILABLE:
        logger.warning("pmdarima no esta instalado. Skipping ARIMA model.")
        return None, None

    logger.info("Entrenando AutoARIMA...")

    model = auto_arima(
        y_train,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        max_order=10,
        trace=False
    )

    logger.info(f"Mejor modelo ARIMA: {model.order}")

    metrics = None
    if y_val is not None:
        n_periods = len(y_val)
        y_pred = model.predict(n_periods=n_periods)
        metrics = calculate_metrics(y_val, y_pred)
        logger.info(f"ARIMA - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, R2: {metrics['r2']:.4f}, MAPE: {metrics['mape']:.2f}%")

    return model, metrics


def train_all_models(data_path: str, output_dir: str = 'models', n_splits: int = 5):
    """
    Entrena todos los modelos (RF, XGBoost, ARIMA) y compara resultados.

    Args:
        data_path: Ruta al CSV con features
        output_dir: Directorio para guardar modelos
        n_splits: Numero de splits para TimeSeriesSplit

    Returns:
        dict con resultados comparativos
    """
    logger.info(f"Cargando datos desde {data_path}")
    df = pd.read_csv(data_path, index_col='Fecha', parse_dates=True)

    feature_cols = [col for col in df.columns if col != 'Valor']
    X = df[feature_cols]
    y = df['Valor']

    logger.info(f"Dataset: {len(X)} filas, {len(feature_cols)} features")

    # Crear directorio de salida
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Time Series Split
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = {
        'random_forest': {'mae': [], 'rmse': [], 'r2': [], 'mape': []},
        'xgboost': {'mae': [], 'rmse': [], 'r2': [], 'mape': []},
        'arima': {'mae': [], 'rmse': [], 'r2': [], 'mape': []}
    }

    # Cross-validation
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold}/{n_splits}")
        logger.info(f"{'='*60}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Random Forest
        _, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val)
        if rf_metrics:
            for key in rf_metrics:
                results['random_forest'][key].append(rf_metrics[key])

        # XGBoost
        _, xgb_metrics = train_xgboost(X_train, y_train, X_val, y_val)
        if xgb_metrics:
            for key in xgb_metrics:
                results['xgboost'][key].append(xgb_metrics[key])

        # ARIMA (solo usa y, no X)
        _, arima_metrics = train_arima(y_train, y_val)
        if arima_metrics:
            for key in arima_metrics:
                results['arima'][key].append(arima_metrics[key])

    # Entrenar modelos finales con todos los datos
    logger.info(f"\n{'='*60}")
    logger.info("Entrenando modelos finales con todos los datos")
    logger.info(f"{'='*60}")

    final_rf, _ = train_random_forest(X, y)
    final_xgb, _ = train_xgboost(X, y)
    final_arima, _ = train_arima(y) if ARIMA_AVAILABLE else (None, None)

    # Guardar modelos
    with open(f"{output_dir}/random_forest.pkl", 'wb') as f:
        pickle.dump(final_rf, f)
    logger.info(f"Random Forest guardado en {output_dir}/random_forest.pkl")

    with open(f"{output_dir}/xgboost.pkl", 'wb') as f:
        pickle.dump(final_xgb, f)
    logger.info(f"XGBoost guardado en {output_dir}/xgboost.pkl")

    if final_arima:
        with open(f"{output_dir}/arima.pkl", 'wb') as f:
            pickle.dump(final_arima, f)
        logger.info(f"ARIMA guardado en {output_dir}/arima.pkl")

    # Guardar feature names
    feature_names_path = Path(output_dir) / 'feature_names.txt'
    with open(feature_names_path, 'w') as f:
        for feature in feature_cols:
            f.write(f"{feature}\n")
    logger.info(f"Feature names guardados en {feature_names_path}")

    # Calcular metricas promedio
    summary = {}
    for model_name, metrics in results.items():
        if metrics['mae']:
            summary[model_name] = {
                'mae_mean': np.mean(metrics['mae']),
                'mae_std': np.std(metrics['mae']),
                'rmse_mean': np.mean(metrics['rmse']),
                'rmse_std': np.std(metrics['rmse']),
                'r2_mean': np.mean(metrics['r2']),
                'r2_std': np.std(metrics['r2']),
                'mape_mean': np.mean(metrics['mape']),
                'mape_std': np.std(metrics['mape'])
            }

    summary['metadata'] = {
        'trained_at': datetime.now().isoformat(),
        'n_features': len(feature_cols),
        'n_samples': len(X),
        'n_splits': n_splits
    }

    # Guardar resumen
    summary_path = Path(output_dir) / 'model_comparison.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Resumen comparativo guardado en {summary_path}")

    return summary


def compare_models(summary):
    """
    Imprime comparacion de modelos.

    Args:
        summary: dict con metricas de todos los modelos
    """
    logger.info(f"\n{'='*80}")
    logger.info("COMPARACION DE MODELOS")
    logger.info(f"{'='*80}\n")

    models = [k for k in summary.keys() if k != 'metadata']

    print(f"{'Modelo':<20} {'MAE':>15} {'RMSE':>15} {'R2':>15} {'MAPE':>15}")
    print(f"{'-'*80}")

    for model_name in models:
        metrics = summary[model_name]
        print(f"{model_name:<20} "
              f"{metrics['mae_mean']:>10.2f} ± {metrics['mae_std']:.2f}   "
              f"{metrics['rmse_mean']:>10.2f} ± {metrics['rmse_std']:.2f}   "
              f"{metrics['r2_mean']:>10.4f} ± {metrics['r2_std']:.4f}   "
              f"{metrics['mape_mean']:>10.2f}% ± {metrics['mape_std']:.2f}%")

    print(f"\n{'='*80}")

    # Determinar mejor modelo (menor MAE)
    best_model = min(models, key=lambda m: summary[m]['mae_mean'])
    print(f"\nMejor modelo segun MAE: {best_model}")
    print(f"  MAE:  {summary[best_model]['mae_mean']:.2f} CLP")
    print(f"  MAPE: {summary[best_model]['mape_mean']:.2f}%")
    print(f"{'='*80}\n")


def train_model(
    data_path: str,
    model_output_path: str,
    n_splits: int = 5,
    save_metrics: bool = True
) -> dict:
    """
    Entrena modelo XGBoost con Time Series Cross-Validation.

    Args:
        data_path: Ruta al CSV con features
        model_output_path: Ruta para guardar modelo entrenado
        n_splits: Número de splits para TimeSeriesSplit
        save_metrics: Si True, guarda métricas en JSON

    Returns:
        dict con métricas de evaluación
    """
    logger.info(f"Cargando datos desde {data_path}")
    df = pd.read_csv(data_path, index_col='Fecha', parse_dates=True)

    # Separar features y target
    feature_cols = [col for col in df.columns if col != 'Valor']
    X = df[feature_cols]
    y = df['Valor']

    logger.info(f"Dataset: {len(X)} filas, {len(feature_cols)} features")
    logger.info(f"Features: {feature_cols[:5]}... (mostrando primeras 5)")

    # Time Series Split
    tscv = TimeSeriesSplit(n_splits=n_splits)

    metrics = {'mae': [], 'rmse': [], 'r2': [], 'mape': []}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        logger.info(f"Training fold {fold}/{n_splits}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Entrenar XGBoost
        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=50
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Evaluar
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        r2 = r2_score(y_val, y_pred)
        mape = (abs((y_val - y_pred) / y_val).mean()) * 100

        metrics['mae'].append(mae)
        metrics['rmse'].append(rmse)
        metrics['r2'].append(r2)
        metrics['mape'].append(mape)

        logger.info(f"Fold {fold} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}, MAPE: {mape:.2f}%")

    # Entrenar modelo final con todos los datos
    logger.info("Entrenando modelo final con todos los datos")
    final_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    final_model.fit(X, y)

    # Guardar modelo
    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_output_path, 'wb') as f:
        pickle.dump(final_model, f)

    logger.info(f"Modelo guardado en {model_output_path}")

    # Guardar nombres de features
    feature_names_path = Path(model_output_path).parent / 'feature_names.txt'
    with open(feature_names_path, 'w') as f:
        for feature in feature_cols:
            f.write(f"{feature}\n")

    logger.info(f"Feature names guardados en {feature_names_path}")

    # Calcular métricas promedio
    avg_metrics = {
        'mae_mean': sum(metrics['mae']) / len(metrics['mae']),
        'mae_std': pd.Series(metrics['mae']).std(),
        'rmse_mean': sum(metrics['rmse']) / len(metrics['rmse']),
        'rmse_std': pd.Series(metrics['rmse']).std(),
        'r2_mean': sum(metrics['r2']) / len(metrics['r2']),
        'r2_std': pd.Series(metrics['r2']).std(),
        'mape_mean': sum(metrics['mape']) / len(metrics['mape']),
        'mape_std': pd.Series(metrics['mape']).std(),
        'trained_at': datetime.now().isoformat(),
        'n_features': len(feature_cols),
        'n_samples': len(X),
        'n_splits': n_splits
    }

    # Guardar métricas
    if save_metrics:
        metrics_path = Path(model_output_path).parent / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(avg_metrics, f, indent=2)
        logger.info(f"Métricas guardadas en {metrics_path}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Métricas promedio (Cross-Validation con {n_splits} folds):")
    logger.info(f"  MAE:  {avg_metrics['mae_mean']:.2f} ± {avg_metrics['mae_std']:.2f}")
    logger.info(f"  RMSE: {avg_metrics['rmse_mean']:.2f} ± {avg_metrics['rmse_std']:.2f}")
    logger.info(f"  R²:   {avg_metrics['r2_mean']:.4f} ± {avg_metrics['r2_std']:.4f}")
    logger.info(f"  MAPE: {avg_metrics['mape_mean']:.2f}% ± {avg_metrics['mape_std']:.2f}%")
    logger.info(f"{'='*60}\n")

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description='Entrenar modelos de predicción USD/CLP')
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/dolar_features.csv',
        help='Ruta al CSV con features'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directorio de salida para modelos'
    )
    parser.add_argument(
        '--n-splits',
        type=int,
        default=5,
        help='Número de splits para Time Series CV'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['all', 'rf', 'xgboost', 'arima'],
        default='all',
        help='Modelo a entrenar (all entrena todos y compara)'
    )

    args = parser.parse_args()

    try:
        if args.model == 'all':
            logger.info("Entrenando todos los modelos y comparando...")
            summary = train_all_models(args.data, args.output_dir, args.n_splits)
            compare_models(summary)
        else:
            logger.info(f"Entrenando solo {args.model}...")
            # Aquí podrías implementar entrenamiento individual si lo necesitas
            logger.warning("Entrenamiento individual aún no implementado. Usa --model all")

        print(f"\nEntrenamiento completado exitosamente!")
        print(f"Modelos guardados en: {args.output_dir}/")

    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
