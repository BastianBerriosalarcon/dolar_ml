# DolarCLP-Predictor

Sistema de predicción del tipo de cambio USD/CLP usando Machine Learning con estrategia multi-modelo.

## Descripción

Proyecto de Machine Learning para predecir el tipo de cambio del dólar estadounidense (USD) frente al peso chileno (CLP) utilizando análisis de series de tiempo multivariadas. Implementa una estrategia comparativa de 3 modelos para seleccionar el mejor predictor.

### Características Principales

- Descarga automática de datos históricos del Banco Central de Chile (30 años)
- Feature engineering avanzado con 23 características técnicas y temporales
- Análisis exploratorio de datos (EDA) completo
- **Estrategia multi-modelo**: Random Forest, XGBoost y ARIMA
- Validación temporal con Time Series Cross-Validation
- Pipeline completa de ML lista para producción

## Estado del Proyecto

- [COMPLETADA] Fase 1: Adquisición de Datos - 10,958 registros descargados
- [COMPLETADA] Fase 2: Feature Engineering - 23 features implementadas, 3,993 registros procesados
- [PENDIENTE] Fase 3: Modelado Multi-Modelo (Random Forest + XGBoost + ARIMA)
- [PENDIENTE] Fase 4: Evaluación y Backtesting
- [PENDIENTE] Fase 5: Despliegue API REST

## Estrategia de Modelado

Este proyecto implementa una **comparación de 3 modelos** para seleccionar el mejor predictor:

| Modelo | Tipo | Propósito |
|--------|------|-----------|
| **Random Forest** | ML Baseline | Robusto, fácil de interpretar |
| **XGBoost** | Gradient Boosting | Alto rendimiento, modelo principal |
| **AutoARIMA** | Estadístico | Baseline clásico de series temporales |

**Criterio de selección**: MAE, RMSE, MAPE y R² en conjunto de test temporal

## Estructura del Proyecto

```
dolar_ml/
├── config/                          # Configuración centralizada
│   ├── config.yaml                 # Parámetros del proyecto
│   ├── logging_config.yaml         # Configuración de logging
│   └── README.md                   # Documentación de configuración
├── data/
│   ├── raw/                        # Datos crudos del BCCh (10,958 registros)
│   └── processed/                  # Datos procesados con features (3,993 registros)
├── logs/                           # Logs de aplicación (ignorado por git)
├── models/                         # Modelos entrenados
│   ├── best_model_xgboost.pkl     # Modelo XGBoost
│   ├── feature_names.txt          # Nombres de features
│   └── metrics.json               # Métricas de evaluación
├── notebooks/
│   ├── 01_data_acquisition.ipynb   # Descarga y validación
│   ├── 02_eda.ipynb                # Análisis exploratorio
│   ├── 03_feature_engineering.ipynb # Documentación de features
│   └── 04_modeling.ipynb           # Entrenamiento de modelos
├── scripts/
│   └── setup.sh                    # Script de configuración
├── src/
│   ├── api/
│   │   └── app.py                  # API REST con FastAPI
│   ├── data/
│   │   ├── fetch_bcch.py           # Extracción API BCCh
│   │   └── preprocess.py           # Procesamiento de datos
│   ├── features/
│   │   └── build_features.py       # Feature engineering completo
│   ├── models/
│   │   ├── train.py                # Entrenamiento de modelos
│   │   └── predict.py              # Predicciones
│   └── utils/
│       ├── config.py               # Carga de configuración
│       ├── logger.py               # Sistema de logging
│       └── metrics.py              # Métricas personalizadas
├── tests/
│   ├── test_data/                  # Tests de módulo data
│   ├── test_features/              # Tests de feature engineering
│   ├── test_models/                # Tests de modelos
│   ├── test_api/                   # Tests de API
│   ├── fixtures/                   # Datos de prueba
│   └── conftest.py                 # Configuración de pytest
├── .dockerignore                   # Archivos a ignorar en Docker
├── .flake8                         # Configuración de flake8
├── .gitignore                      # Archivos a ignorar en git
├── CONTRIBUTING.md                 # Guía de contribución
├── Dockerfile                      # Imagen Docker
├── docker-compose.yml              # Orquestación de contenedores
├── download_data.py                # Script principal de descarga
├── LICENSE                         # Licencia del proyecto
├── pyproject.toml                  # Configuración de herramientas Python
├── requirements.txt                # Dependencias de producción
├── requirements-dev.txt            # Dependencias de desarrollo
└── README.md                       # Este archivo
```

## Instalación

### Prerrequisitos

- Python 3.8+
- Credenciales del Banco Central de Chile

### Setup

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/dolar_ml.git
cd dolar_ml

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar credenciales (variables de entorno)
export BC_USER="tu_email@ejemplo.com"
export BC_PASS="tu_contraseña"
```

## Uso

### 1. Descargar datos

```bash
python download_data.py
```

### 2. Generar features

```bash
python -m src.features.build_features
```

### 3. Análisis exploratorio

```bash
jupyter lab
```

## Dataset

- **Fuente**: Banco Central de Chile
- **Serie**: F073.TCO.PRE.Z.D (Dólar observado diario)
- **Período**: 1995-2025 (30 años)
- **Registros totales**: 10,958
- **Registros procesados**: 3,993 (con features completas)

## Features Implementadas (23 características)

### Variables Lag (3)
- `lag_1`, `lag_7`, `lag_30`: Valores históricos desplazados

### Promedios Móviles (3)
- `ma_7`, `ma_30`, `ma_90`: Suavizan tendencias de corto, medio y largo plazo

### Features Temporales (4)
- `day_of_week`, `month`, `quarter`, `year`: Capturan estacionalidad

### Indicadores Técnicos (5)
- `roc_7`, `roc_30`: Rate of Change
- `momentum_7`, `momentum_30`: Momentum
- `daily_return`: Retornos diarios porcentuales

### Volatilidad (3)
- `volatility_7`, `volatility_30`: Desviación estándar móvil
- `range_7`: Rango de precio (max-min)

### Objetivo
- `Valor`: Tipo de cambio USD/CLP (variable target)

## Stack Tecnológico

### Core
- **Python 3.12**
- **Pandas** & **NumPy**: Manipulación de datos
- **Requests**: API del Banco Central de Chile

### Machine Learning
- **Scikit-learn**: Preprocesamiento, métricas, validación, Random Forest
- **XGBoost**: Gradient boosting avanzado
- **pmdarima**: AutoARIMA para series temporales
- **statsmodels**: Análisis estadístico (ADF test)

### Visualización
- **Matplotlib** & **Seaborn**: Gráficos estáticos
- **Plotly**: Gráficos interactivos (recomendado)

### Desarrollo
- **JupyterLab**: Notebooks interactivos
- **Git**: Control de versiones

## Métricas de Éxito Esperadas

### Técnicas
- **MAPE < 2%**: Excelente
- **MAPE 2-5%**: Bueno
- **MAPE > 5%**: Requiere mejoras

### De Negocio
- Precisión en predicciones a 1 día: > 95%
- Precisión en predicciones a 7 días: > 85%
- Tiempo de inferencia: < 100ms

## Próximos Pasos

1. Implementar los 3 modelos comparativos
2. Crear notebook `04_modeling.ipynb` con comparación
3. Seleccionar modelo ganador según métricas
4. Implementar backtesting con walk-forward validation
5. Desarrollar API REST con FastAPI
6. Dockerizar para despliegue

## Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/NuevaFeature`)
3. Commit tus cambios (`git commit -m 'Agregar NuevaFeature'`)
4. Push a la rama (`git push origin feature/NuevaFeature`)
5. Abre un Pull Request

**Estándares de código**:
- NO usar emojis en código, comentarios, prints o notebooks
- Seguir convenciones PEP 8
- Documentar funciones con docstrings

## Licencia

Por definir

## Autor

**Bastián Berríos**
Email: bastianberrios.a@gmail.com

## Agradecimientos

- **Banco Central de Chile** por proporcionar la API de datos históricos
- Comunidad open-source de Python y bibliotecas de ML
