# Configuración del Proyecto

Esta carpeta contiene los archivos de configuración centralizados del proyecto.

## Archivos

### `config.yaml`
Configuración principal del proyecto:
- Parámetros del modelo XGBoost
- Configuración de cross-validation
- Definición de features a crear
- Paths de archivos y directorios
- Configuración de la API

### `logging_config.yaml`
Configuración del sistema de logging:
- Formateadores de logs
- Handlers (consola, archivo, errores)
- Niveles de logging por módulo
- Rotación de archivos de log

## Uso

Para cargar la configuración en tu código:

```python
import yaml

# Cargar configuración principal
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Acceder a parámetros
model_params = config["model"]["params"]
n_splits = config["cross_validation"]["n_splits"]
```

Para configurar logging:

```python
import logging.config
import yaml

# Cargar configuración de logging
with open("config/logging_config.yaml", "r") as f:
    log_config = yaml.safe_load(f)

logging.config.dictConfig(log_config)

# Usar logger
logger = logging.getLogger(__name__)
logger.info("Logger configurado correctamente")
```

## Notas

- Los archivos YAML usan sintaxis estándar YAML
- Los paths son relativos a la raíz del proyecto
- Los logs se guardan en la carpeta `logs/` (ignorada por git)
