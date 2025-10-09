"""
Script de entrenamiento de modelos con logging y versionado.
"""
import argparse
import logging
import pickle
from pathlib import Path
from datetime import datetime
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import json

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    parser = argparse.ArgumentParser(description='Entrenar modelo de predicción USD/CLP')
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/dolar_features.csv',
        help='Ruta al CSV con features'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/best_model_xgboost.pkl',
        help='Ruta de salida para el modelo'
    )
    parser.add_argument(
        '--n-splits',
        type=int,
        default=5,
        help='Número de splits para Time Series CV'
    )

    args = parser.parse_args()

    try:
        metrics = train_model(args.data, args.output, args.n_splits)

        print(f"\nEntrenamiento completado exitosamente!")
        print(f"\nResultados:")
        print(f"   MAE:  {metrics['mae_mean']:.2f} CLP")
        print(f"   RMSE: {metrics['rmse_mean']:.2f} CLP")
        print(f"   R²:   {metrics['r2_mean']:.4f}")
        print(f"   MAPE: {metrics['mape_mean']:.2f}%")
        print(f"\nModelo guardado en: {args.output}")

    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
