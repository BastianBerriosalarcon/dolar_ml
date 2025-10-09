"""
API REST para servir predicciones del modelo de dólar.
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pickle
import logging
from datetime import datetime
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="DolarCLP Predictor API",
    description="API para predicción de tipo de cambio USD/CLP usando XGBoost",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo al inicio
MODEL_PATH = "models/best_model_xgboost.pkl"
FEATURES_PATH = "models/feature_names.txt"

model = None
feature_names = []

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(FEATURES_PATH, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    logger.info(f"Modelo cargado exitosamente. Features: {len(feature_names)}")
except FileNotFoundError as e:
    logger.error(f"Error cargando modelo: {e}")
    logger.error("Asegúrate de entrenar el modelo primero: python -m src.models.train")
except Exception as e:
    logger.error(f"Error inesperado cargando modelo: {e}")


# Schemas de entrada/salida
class PredictionInput(BaseModel):
    """Schema para request de predicción."""
    features: Dict[str, float] = Field(
        ...,
        description="Diccionario con nombres y valores de features"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "Valor_lag_1": 920.5,
                    "Valor_lag_7": 918.2,
                    "Valor_lag_30": 915.0,
                    "MA_7": 919.0,
                    "MA_30": 917.5,
                    "MA_90": 910.0,
                    "day_of_week": 1,
                    "month": 3,
                    "quarter": 1,
                    "ROC_7": 0.25,
                    "momentum_7": 2.3
                }
            }
        }


class PredictionOutput(BaseModel):
    """Schema para response de predicción."""
    prediction: float = Field(..., description="Valor predicho USD/CLP")
    timestamp: str = Field(..., description="Timestamp de la predicción")
    model_version: str = Field(default="1.0.0", description="Versión del modelo")


class HealthResponse(BaseModel):
    """Schema para health check."""
    status: str
    model_loaded: bool
    features_count: int
    timestamp: str


class FeatureInfo(BaseModel):
    """Schema para información de features."""
    name: str
    description: str


# Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Endpoint raíz con información básica de la API."""
    return {
        "message": "DolarCLP Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint para monitoreo."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        features_count=len(feature_names),
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionOutput, status_code=status.HTTP_200_OK)
async def predict(input_data: PredictionInput):
    """
    Realiza predicción del tipo de cambio USD/CLP.

    Args:
        input_data: Features para predicción

    Returns:
        Predicción con timestamp y metadata

    Raises:
        HTTPException: Si el modelo no está cargado o hay error en predicción
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible. Entrena el modelo primero: python -m src.models.train"
        )

    try:
        # Preparar features en orden correcto
        import pandas as pd

        # Validar que estén todas las features
        missing = set(feature_names) - set(input_data.features.keys())
        if missing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Faltan features requeridas: {list(missing)}"
            )

        df = pd.DataFrame([input_data.features])[feature_names]

        # Predecir
        prediction = float(model.predict(df)[0])

        logger.info(f"Predicción exitosa: {prediction:.2f} CLP/USD")

        return PredictionOutput(
            prediction=prediction,
            timestamp=datetime.now().isoformat(),
            model_version="1.0.0"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al generar predicción: {str(e)}"
        )


@app.post("/predict/batch", response_model=List[PredictionOutput])
async def predict_batch(inputs: List[PredictionInput]):
    """
    Realiza predicciones en batch para múltiples observaciones.

    Args:
        inputs: Lista de features para predicción

    Returns:
        Lista de predicciones
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible"
        )

    predictions = []
    for input_data in inputs:
        result = await predict(input_data)
        predictions.append(result)

    return predictions


@app.get("/features", response_model=Dict[str, any])
async def get_features():
    """
    Retorna la lista de features requeridas por el modelo.

    Returns:
        Diccionario con nombres de features y descripciones
    """
    feature_descriptions = {
        "Valor_lag_1": "Valor del dólar hace 1 día",
        "Valor_lag_7": "Valor del dólar hace 7 días",
        "Valor_lag_30": "Valor del dólar hace 30 días",
        "MA_7": "Media móvil de 7 días",
        "MA_30": "Media móvil de 30 días",
        "MA_90": "Media móvil de 90 días",
        "day_of_week": "Día de la semana (0=Lunes, 6=Domingo)",
        "day_of_month": "Día del mes (1-31)",
        "month": "Mes (1-12)",
        "quarter": "Trimestre (1-4)",
        "year": "Año",
        "is_month_start": "Inicio de mes (0 o 1)",
        "is_month_end": "Fin de mes (0 o 1)",
        "ROC_7": "Rate of Change 7 días (%)",
        "ROC_30": "Rate of Change 30 días (%)",
        "momentum_7": "Momentum 7 días",
        "momentum_30": "Momentum 30 días",
        "daily_return": "Retorno diario (%)",
        "volatility_7": "Volatilidad 7 días",
        "volatility_30": "Volatilidad 30 días",
        "range_7": "Rango 7 días (max-min)",
        "range_30": "Rango 30 días (max-min)"
    }

    features_info = []
    for feature_name in feature_names:
        features_info.append({
            "name": feature_name,
            "description": feature_descriptions.get(feature_name, "Feature técnica")
        })

    return {
        "features": features_info,
        "total_count": len(feature_names)
    }


@app.get("/metrics", response_model=Dict[str, any])
async def get_metrics():
    """
    Retorna métricas de evaluación del modelo.

    Returns:
        Diccionario con métricas del modelo
    """
    metrics_path = Path(MODEL_PATH).parent / "metrics.json"

    if not metrics_path.exists():
        return {
            "message": "Métricas no disponibles. Entrena el modelo primero.",
            "mae": None,
            "rmse": None,
            "r2": None
        }

    import json
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    return metrics


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
