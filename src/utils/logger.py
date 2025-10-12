"""
Configuración centralizada de logging para el proyecto.
Incluye logging estructurado para producción con formato JSON.
"""

import logging
import logging.config
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


def setup_logging(
    config_path: str = "config/logging_config.yaml",
    default_level: int = logging.INFO
) -> None:
    """
    Configura el sistema de logging usando archivo YAML.

    Args:
        config_path: Ruta al archivo de configuración de logging
        default_level: Nivel de logging por defecto si falla la carga
    """
    config_file = Path(config_path)

    # Crear directorio de logs si no existe
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                log_config = yaml.safe_load(f)
            logging.config.dictConfig(log_config)
        except Exception as e:
            logging.basicConfig(level=default_level)
            logging.error(f"Error al cargar configuración de logging: {e}")
    else:
        logging.basicConfig(level=default_level)
        logging.warning(f"Archivo de configuración no encontrado: {config_path}")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Obtiene un logger configurado.

    Args:
        name: Nombre del logger (usualmente __name__)

    Returns:
        Instancia de Logger
    """
    return logging.getLogger(name)


class JSONFormatter(logging.Formatter):
    """
    Formateador JSON para logging estructurado en producción.
    Útil para sistemas de agregación de logs (ELK, Splunk, etc.)
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Formatea el log record como JSON.

        Args:
            record: LogRecord a formatear

        Returns:
            String JSON con la información del log
        """
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        if hasattr(record, 'extra'):
            log_data['extra'] = record.extra

        return json.dumps(log_data)


def setup_production_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configura logging para producción con formato JSON.

    Args:
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Archivo de log (opcional). Si no se especifica, solo console output
    """
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)

    handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(JSONFormatter())
    handlers.append(console_handler)

    if log_file:
        file_handler = logging.FileHandler(logs_dir / log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(JSONFormatter())
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        handlers=handlers
    )


def setup_simple_logging(log_level: str = "INFO") -> None:
    """
    Configura logging simple para desarrollo.

    Args:
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def log_prediction(logger: logging.Logger, model_name: str, prediction: float, features: Dict[str, Any]) -> None:
    """
    Log especializado para predicciones.

    Args:
        logger: Instancia de Logger
        model_name: Nombre del modelo usado
        prediction: Valor predicho
        features: Features usadas en la predicción
    """
    logger.info(
        "Predicción realizada",
        extra={
            'event_type': 'prediction',
            'model': model_name,
            'prediction': prediction,
            'features': features,
            'timestamp': datetime.utcnow().isoformat()
        }
    )


def log_model_metrics(logger: logging.Logger, model_name: str, metrics: Dict[str, float]) -> None:
    """
    Log especializado para métricas de modelo.

    Args:
        logger: Instancia de Logger
        model_name: Nombre del modelo
        metrics: Dict con métricas (MAE, RMSE, etc.)
    """
    logger.info(
        f"Métricas del modelo {model_name}",
        extra={
            'event_type': 'model_metrics',
            'model': model_name,
            'metrics': metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
    )