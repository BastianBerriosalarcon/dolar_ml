"""
Métricas personalizadas para evaluación de modelos.
"""

import numpy as np
from typing import Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calcula métricas de evaluación para regresión.

    Args:
        y_true: Valores reales
        y_pred: Valores predichos

    Returns:
        Diccionario con las métricas calculadas
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Median Absolute Error
    median_ae = np.median(np.abs(y_true - y_pred))

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape),
        "median_ae": float(median_ae)
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Formatea las métricas para impresión legible.

    Args:
        metrics: Diccionario con métricas

    Returns:
        String formateado con las métricas
    """
    lines = ["Métricas de Evaluación:"]
    lines.append("-" * 40)

    metric_names = {
        "mae": "MAE (Mean Absolute Error)",
        "mse": "MSE (Mean Squared Error)",
        "rmse": "RMSE (Root Mean Squared Error)",
        "r2": "R² Score",
        "mape": "MAPE (%)",
        "median_ae": "Median Absolute Error"
    }

    for key, value in metrics.items():
        name = metric_names.get(key, key)
        if key == "r2":
            lines.append(f"{name:30s}: {value:.4f}")
        elif key == "mape":
            lines.append(f"{name:30s}: {value:.2f}%")
        else:
            lines.append(f"{name:30s}: {value:.2f}")

    return "\n".join(lines)


def save_metrics(metrics: Dict[str, Any], output_path: str) -> None:
    """
    Guarda las métricas en un archivo JSON.

    Args:
        metrics: Diccionario con métricas
        output_path: Ruta del archivo de salida
    """
    import json
    from pathlib import Path

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)