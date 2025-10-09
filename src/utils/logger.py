"""
Configuración centralizada de logging para el proyecto.
"""

import logging
import logging.config
import yaml
from pathlib import Path
from typing import Optional


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