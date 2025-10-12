"""
Script de inferencia para predicciones en producción.
Incluye funciones para predicción simple, en rango, y carga de modelos.
"""
import argparse
import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
from typing import Dict, Union, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DolarPredictor:
    """Predictor de tipo de cambio USD/CLP."""

    def __init__(self, model_path: str, feature_names_path: str = None):
        """
        Inicializa el predictor cargando modelo y feature names.

        Args:
            model_path: Ruta al modelo serializado
            feature_names_path: Ruta al archivo con nombres de features (opcional)
        """
        logger.info(f"Cargando modelo desde {model_path}")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Si no se proporciona ruta de features, buscar en el mismo directorio del modelo
        if feature_names_path is None:
            feature_names_path = Path(model_path).parent / 'feature_names.txt'

        if Path(feature_names_path).exists():
            logger.info(f"Cargando feature names desde {feature_names_path}")
            with open(feature_names_path, 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
        else:
            logger.warning(f"No se encontró {feature_names_path}, usando features del modelo")
            self.feature_names = None

        logger.info(f"Predictor listo. Features: {len(self.feature_names) if self.feature_names else 'N/A'}")

    def predict(self, features: Union[Dict, pd.DataFrame]) -> Union[float, pd.Series]:
        """
        Realiza predicción para un conjunto de features.

        Args:
            features: Dict con nombres y valores de features o DataFrame

        Returns:
            Predicción del valor USD/CLP (float si input es dict, Series si es DataFrame)

        Raises:
            ValueError: Si faltan features requeridas
        """
        # Convertir dict a DataFrame si es necesario
        if isinstance(features, dict):
            df = pd.DataFrame([features])
            return_single = True
        else:
            df = features.copy()
            return_single = False

        # Validar features si tenemos la lista
        if self.feature_names:
            missing = set(self.feature_names) - set(df.columns)
            if missing:
                raise ValueError(f"Faltan features requeridas: {missing}")

            # Reordenar columnas según orden esperado
            df = df[self.feature_names]

        # Predecir
        predictions = self.model.predict(df)

        if return_single:
            prediction = float(predictions[0])
            logger.info(f"Predicción: {prediction:.2f} CLP/USD")
            return prediction
        else:
            logger.info(f"Predicciones completadas: {len(predictions)} valores")
            return pd.Series(predictions, index=df.index)

    def predict_from_csv(self, csv_path: str, output_path: str = None) -> pd.DataFrame:
        """
        Predice para múltiples observaciones desde CSV.

        Args:
            csv_path: Ruta al CSV con features
            output_path: Ruta para guardar predicciones (opcional)

        Returns:
            DataFrame con predicciones
        """
        logger.info(f"Cargando datos desde {csv_path}")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

        # Validar columnas
        if self.feature_names:
            missing = set(self.feature_names) - set(df.columns)
            if missing:
                raise ValueError(f"Faltan columnas: {missing}")

        # Predecir
        predictions = self.predict(df)

        result = df.copy()
        result['prediction'] = predictions

        logger.info(f"Predicciones completadas: {len(result)} filas")

        # Guardar si se especifica output
        if output_path:
            result.to_csv(output_path)
            logger.info(f"Predicciones guardadas en {output_path}")

        return result

    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Obtiene la importancia de features del modelo.

        Args:
            top_n: Número de features más importantes a retornar

        Returns:
            DataFrame con features e importancias
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_

            if self.feature_names:
                feature_imp = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
            else:
                feature_imp = pd.DataFrame({
                    'feature': [f'feature_{i}' for i in range(len(importances))],
                    'importance': importances
                }).sort_values('importance', ascending=False)

            return feature_imp.head(top_n)
        else:
            logger.warning("El modelo no tiene feature_importances_")
            return pd.DataFrame()


def load_model(model_path: str) -> DolarPredictor:
    """
    Carga un modelo entrenado.

    Args:
        model_path: Ruta al archivo del modelo (.pkl)

    Returns:
        Instancia de DolarPredictor

    Example:
        predictor = load_model('models/xgboost.pkl')
    """
    return DolarPredictor(model_path)


def predict_single(model_path: str, features: Dict) -> Tuple[float, Dict]:
    """
    Realiza una predicción simple con features proporcionadas.

    Args:
        model_path: Ruta al modelo entrenado
        features: Dict con features para predicción

    Returns:
        Tuple con (predicción, metadata)

    Example:
        prediction, metadata = predict_single(
            'models/xgboost.pkl',
            {'lag_1': 950.5, 'ma_7': 945.2, ...}
        )
    """
    predictor = load_model(model_path)
    prediction = predictor.predict(features)

    metadata = {
        'prediction': prediction,
        'model': Path(model_path).name,
        'timestamp': datetime.now().isoformat(),
        'features_used': list(features.keys())
    }

    return prediction, metadata


def predict_range(model_path: str, features_df: pd.DataFrame, n_days: int = 7) -> pd.DataFrame:
    """
    Realiza predicciones para un rango de días.

    NOTA: Esta es una versión simplificada. Para predicciones multi-step reales
    necesitarías implementar rolling predictions con features autogeneradas.

    Args:
        model_path: Ruta al modelo entrenado
        features_df: DataFrame con features para cada día
        n_days: Número de días a predecir

    Returns:
        DataFrame con predicciones

    Example:
        predictions_df = predict_range(
            'models/xgboost.pkl',
            features_df,
            n_days=7
        )
    """
    predictor = load_model(model_path)

    # Predecir para las primeras n_days filas
    subset = features_df.head(n_days)
    predictions = predictor.predict(subset)

    result = pd.DataFrame({
        'date': subset.index,
        'prediction': predictions,
        'model': Path(model_path).name
    })

    logger.info(f"Predicciones generadas para {n_days} días")
    return result


def compare_models_predictions(model_paths: List[str], features: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
    """
    Compara predicciones de múltiples modelos.

    Args:
        model_paths: Lista de rutas a modelos entrenados
        features: Dict o DataFrame con features

    Returns:
        DataFrame con predicciones de cada modelo

    Example:
        comparison = compare_models_predictions(
            ['models/random_forest.pkl', 'models/xgboost.pkl'],
            features_dict
        )
    """
    results = {}

    for model_path in model_paths:
        model_name = Path(model_path).stem
        try:
            predictor = load_model(model_path)
            prediction = predictor.predict(features)

            if isinstance(prediction, pd.Series):
                results[model_name] = prediction
            else:
                results[model_name] = [prediction]

            logger.info(f"Predicción de {model_name}: {prediction if not isinstance(prediction, pd.Series) else prediction.mean():.2f}")

        except Exception as e:
            logger.error(f"Error con modelo {model_name}: {e}")
            results[model_name] = None

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Predicción de USD/CLP con modelo entrenado')
    parser.add_argument(
        '--model',
        type=str,
        default='models/best_model_xgboost.pkl',
        help='Ruta al modelo entrenado'
    )
    parser.add_argument(
        '--features',
        type=str,
        help='Ruta al archivo de feature names (opcional)'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='CSV con features para predicción'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='CSV para guardar predicciones (opcional)'
    )
    parser.add_argument(
        '--show-importance',
        action='store_true',
        help='Mostrar feature importance'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Número de features más importantes a mostrar'
    )

    args = parser.parse_args()

    try:
        # Inicializar predictor
        predictor = DolarPredictor(args.model, args.features)

        # Mostrar feature importance si se solicita
        if args.show_importance:
            print("\nFeature Importance (Top {}):".format(args.top_n))
            print("=" * 50)
            importance_df = predictor.get_feature_importance(args.top_n)
            if not importance_df.empty:
                for idx, row in importance_df.iterrows():
                    print(f"  {row['feature']:30s} {row['importance']:.4f}")
            print("=" * 50 + "\n")

        # Predecir
        results = predictor.predict_from_csv(args.input, args.output)

        # Mostrar resumen
        print(f"\nPredicciones completadas!")
        print(f"   Total de predicciones: {len(results)}")
        print(f"   Rango: {results['prediction'].min():.2f} - {results['prediction'].max():.2f} CLP")
        print(f"   Promedio: {results['prediction'].mean():.2f} CLP")

        if args.output:
            print(f"\nResultados guardados en: {args.output}")
        else:
            print("\nPrimeras 10 predicciones:")
            print(results[['prediction']].head(10))

    except Exception as e:
        logger.error(f"Error durante la predicción: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
