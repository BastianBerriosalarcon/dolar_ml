"""
Utilidades para cargar y gestionar configuración del proyecto.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Carga el archivo de configuración YAML.

    Args:
        config_path: Ruta al archivo de configuración

    Returns:
        Diccionario con la configuración

    Raises:
        FileNotFoundError: Si el archivo no existe
        yaml.YAMLError: Si hay error al parsear el YAML
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_model_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrae los parámetros del modelo de la configuración.

    Args:
        config: Diccionario de configuración completo

    Returns:
        Diccionario con parámetros del modelo
    """
    return config.get("model", {}).get("params", {})


def get_data_paths(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Extrae las rutas de datos de la configuración.

    Args:
        config: Diccionario de configuración completo

    Returns:
        Diccionario con rutas de datos
    """
    return config.get("data", {})


def get_cv_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrae la configuración de cross-validation.

    Args:
        config: Diccionario de configuración completo

    Returns:
        Diccionario con configuración de CV
    """
    return config.get("cross_validation", {})