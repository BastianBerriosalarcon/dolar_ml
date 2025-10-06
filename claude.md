# DólarCLP-Predictor: Sistema de Predicción del Tipo de Cambio USD/CLP

## Estándares de Código

**IMPORTANTE**: Todo el código en este proyecto debe seguir estos estándares:
- **NO usar emojis** en código, comentarios, prints, logs o notebooks
- Mantener un estilo profesional y limpio
- Usar mensajes claros y concisos sin caracteres decorativos

## Visión General del Proyecto

Este proyecto implementa un **modelo predictivo de Machine Learning** para pronosticar el tipo de cambio del dólar estadounidense (USD) frente al peso chileno (CLP). El sistema utiliza análisis de series de tiempo multivariadas, combinando datos históricos del Banco Central de Chile con variables económicas relevantes para generar predicciones precisas.

## Objetivos del Proyecto

### Objetivo Principal
Desarrollar un modelo de series de tiempo multivariado capaz de:
- Aprender patrones complejos de datos históricos (30 años de información)
- Generar predicciones precisas del valor futuro USD/CLP
- Adaptarse a cambios en el comportamiento del mercado

### Objetivos Secundarios
- Crear una pipeline automatizada de adquisición y procesamiento de datos
- Implementar ingeniería de características robusta
- Proporcionar métricas de rendimiento confiables
- Desarrollar un endpoint simulado para servir predicciones

## Roadmap de Implementación

### [COMPLETADA] Fase 1: Adquisición de Datos (COMPLETADA)
- **Extracción de datos del Banco Central de Chile (BCCh)**
  - Serie F073.TCO.PRE.Z.D (Dólar observado diario)
  - Período: Últimos 30 años (1995-2025)
  - Total de observaciones: ~10,951 registros
  - Valores válidos: ~7,471 (68.2%)
  - Valores faltantes (ND): ~3,480 (31.8%)
- **Consolidación y limpieza**
  - Formato de salida: CSV con columnas [Fecha, Valor, statusCode]
  - Manejo de valores nulos y días sin operación

### [COMPLETADA] Fase 2: Ingeniería de Características (COMPLETADA)
- **Variables lag**: Valores históricos desplazados (t-1, t-7, t-30)
- **Promedios móviles**: MA(7), MA(30), MA(90)
- **Indicadores de tendencia**: ROC (Rate of Change), momentum
- **Variables temporales**: día de semana, mes, trimestre
- **Volatilidad**: Desviación estándar móvil

### [PENDIENTE] Fase 3: Modelado y Entrenamiento (PENDIENTE)

**Estrategia Multi-Modelo**: Implementar 3 modelos y seleccionar el mejor según métricas

#### Modelo 1: Random Forest (Baseline ML)
- **Propósito**: Modelo baseline de árbol, fácil de interpretar
- **Ventajas**:
  - Robusto ante overfitting
  - No requiere normalización de features
  - Feature importance clara
  - Pocos hiperparámetros críticos
- **Hiperparámetros a optimizar**:
  - `n_estimators`: 100, 200, 500
  - `max_depth`: 10, 20, 30, None
  - `min_samples_split`: 2, 5, 10
  - `max_features`: 'sqrt', 'log2'
- **Librería**: `sklearn.ensemble.RandomForestRegressor`

#### Modelo 2: XGBoost (Modelo Principal)
- **Propósito**: Modelo avanzado de gradient boosting
- **Ventajas**:
  - Excelente rendimiento en series de tiempo
  - Manejo nativo de valores faltantes
  - Robustez ante outliers
  - Capacidad de capturar relaciones no lineales complejas
- **Hiperparámetros a optimizar**:
  - `learning_rate`: 0.01, 0.05, 0.1
  - `max_depth`: 3, 5, 7, 10
  - `n_estimators`: 100, 500, 1000
  - `subsample`: 0.7, 0.8, 0.9
  - `colsample_bytree`: 0.7, 0.8, 0.9
- **Librería**: `xgboost.XGBRegressor`

#### Modelo 3: ARIMA/AutoARIMA (Baseline Estadístico)
- **Propósito**: Modelo estadístico clásico para comparación
- **Ventajas**:
  - Captura autocorrelación temporal naturalmente
  - No requiere feature engineering
  - Muy interpretable (componentes AR, I, MA)
  - Baseline tradicional en forecasting financiero
- **Configuración**:
  - AutoARIMA para búsqueda automática de (p,d,q)
  - Prueba de estacionaridad (ADF test)
  - Solo usa serie temporal univariada
- **Librería**: `pmdarima.auto_arima`

#### Configuración General
- **Train/Test split**: 80/20 (manteniendo orden temporal)
- **Validación**: Time Series Cross-Validation (TimeSeriesSplit)
- **Optimización**: GridSearchCV o RandomizedSearchCV
- **Métricas de selección**:
  - MAE (Mean Absolute Error) - métrica principal
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)
  - R² Score

#### Estrategia de Comparación
1. Entrenar los 3 modelos con mismos datos de entrenamiento
2. Evaluar en mismo conjunto de test
3. Comparar métricas en tabla comparativa
4. Analizar feature importance (RF y XGBoost)
5. **Seleccionar modelo ganador** o implementar ensemble
6. Opcional: Combinar predicciones (promedio ponderado o stacking)

### [PENDIENTE] Fase 4: Evaluación y Backtesting (PENDIENTE)
- **Métricas de evaluación**:
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)
  - R² Score
- **Backtesting**:
  - Walk-forward validation
  - Pruebas en períodos de alta volatilidad
  - Análisis de residuos

### [PENDIENTE] Fase 5: Despliegue (PENDIENTE)
- **API REST** con FastAPI o Flask
- **Endpoints**:
  - `/predict`: Predicción del próximo valor
  - `/predict/range`: Predicción para N días
  - `/metrics`: Métricas de rendimiento del modelo
- **Dockerización** para portabilidad

## Arquitectura de Datos

### Fuentes de Datos Actuales
1. **Banco Central de Chile (BCCh)**
   - `USD/CLP`: Tipo de cambio observado diario
   - `TPM`: Tasa de Política Monetaria (planificado)

2. **Mercado de Commodities** (planificado)
   - `Cobre`: Precio internacional (correlación alta con economía chilena)

### Fuentes Adicionales Recomendadas
3. **Indicadores Macroeconómicos**
   - Inflación (IPC)
   - PIB trimestral
   - Balanza comercial
   - Reservas internacionales

4. **Mercado Financiero Internacional**
   - DXY (US Dollar Index)
   - VIX (Volatility Index)
   - Spreads de bonos soberanos

5. **Sentiment Analysis** (opcional)
   - Noticias económicas
   - Indicadores de confianza empresarial/consumidor

## Stack Tecnológico

### Lenguaje y Librerías Core
- **Python 3.12** (ambiente actual)
- **Pandas**: Manipulación y análisis de datos
- **NumPy**: Operaciones numéricas
- **Requests**: Adquisición de datos vía API

### Machine Learning
- **Scikit-learn**: Preprocesamiento, métricas, validación, Random Forest
- **XGBoost**: Modelo de gradient boosting avanzado
- **pmdarima**: AutoARIMA para modelos estadísticos de series temporales
- **Optuna** (opcional): Optimización avanzada de hiperparámetros

### Visualización
- **Matplotlib**: Gráficos estáticos
- **Plotly** (recomendado): Gráficos interactivos
- **Seaborn**: Análisis exploratorio de datos

### Entorno de Desarrollo
- **JupyterLab**: Prototipado y análisis interactivo
- **VSCode/PyCharm**: Desarrollo de scripts
- **Git**: Control de versiones

## Instalación y Configuración

### Prerrequisitos
- Python 3.8+
- pip
- Credenciales del Banco Central de Chile

### Setup en Linux/WSL

```bash
# 1. Clonar el repositorio (si aplica)
git clone <repo-url>
cd dolar_ml

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno (recomendado)
export BC_USER="tu_email@ejemplo.com"
export BC_PASS="tu_contraseña"
export DOLAR_VERBOSE=0  # 1 para output detallado

# 5. Ejecutar JupyterLab para análisis de datos
jupyter lab
```

## Estructura del Proyecto

### Estado Actual (Implementado)

```
dolar_ml/
├── .gitignore           # Configuración Git (excluye venv, __pycache__, .env)
├── claude.md            # Este archivo - Documentación del proyecto
├── requirements.txt     # Dependencias completas
├── download_data.py     # Script para descarga de datos
├── data/
│   ├── raw/             # Datos crudos del BCCh
│   │   └── dolar_bcch.csv   # 10,958 registros (1995-2025)
│   └── processed/       # Datos procesados para ML
│       └── dolar_features.csv   # 3,993 registros con 23 features
├── notebooks/
│   ├── 01_data_acquisition.ipynb   # Descarga y validación
│   └── 02_eda.ipynb                # Análisis exploratorio
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── fetch_bcch.py    # [COMPLETADA] Extracción API BCCh
│   │   └── preprocess.py    # [COMPLETADA] Conversión JSON→DataFrame→CSV
│   └── features/
│       ├── __init__.py
│       └── build_features.py   # [COMPLETADA] Feature engineering completo
└── venv/                # Entorno virtual Python (NO se versiona)
```

**Descripción de módulos implementados:**

- **[src/data/fetch_bcch.py](src/data/fetch_bcch.py)**:
  - Funciones: `fetch_dolar_observado()`, `fetch_series()`, `get_bcch_credentials()`
  - Descarga series temporales desde API REST del BCCh
  - Requiere variables de entorno: `BC_USER`, `BC_PASS`
  - Serie principal: F073.TCO.PRE.Z.D (USD/CLP observado diario)
  - Timeout de 30 segundos, manejo de años bisiestos

- **[src/data/preprocess.py](src/data/preprocess.py)**:
  - Funciones: `json_to_dataframe()`, `save_to_csv()`, `process_and_save()`, `print_summary()`
  - Transforma respuesta JSON de BCCh a DataFrame de Pandas
  - Maneja valores faltantes (ND) y conversión de tipos
  - Guarda datos en `data/raw/dolar_bcch.csv`

- **[src/features/build_features.py](src/features/build_features.py)**:
  - Funciones: `add_lag_features()`, `add_moving_averages()`, `add_temporal_features()`, `add_technical_indicators()`, `add_volatility_features()`, `build_all_features()`, `prepare_for_ml()`
  - Features implementadas:
    - Lags: t-1, t-7, t-30
    - Moving Averages: 7, 30, 90 días
    - Temporales: día semana, mes, trimestre, año
    - Técnicas: ROC(7,30), Momentum(7,30), retornos diarios
    - Volatilidad: desviación estándar móvil (7,30), rangos
  - Dataset final: 3,993 registros con 23 features

### Estructura Objetivo (Planificada)

```
dolar_ml/
├── data/
│   ├── raw/              # Datos crudos descargados
│   │   └── dolar_bcch.csv
│   ├── processed/        # Datos con features engineering
│   └── external/         # Fuentes adicionales (cobre, TPM, etc.)
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb
├── src/
│   ├── data/
│   │   ├── fetch_bcch.py    # [COMPLETADA] IMPLEMENTADO
│   │   └── preprocess.py    # [COMPLETADA] IMPLEMENTADO
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train.py
│   │   └── predict.py
│   └── api/
│       └── app.py
├── models/               # Modelos entrenados serializados
├── tests/
├── requirements.txt
├── README.md
└── .env                  # Variables de entorno (no versionar)
```

## Consideraciones Importantes

### 1. Calidad de Datos
- **31.8% de valores faltantes (ND)**: Corresponden a fines de semana y feriados
- **Estrategias de imputación**:
  - Forward fill para días consecutivos
  - Interpolación lineal para gaps cortos
  - No imputar si gap > 5 días (mantener como NaN)

### 2. Validación Temporal
- **NUNCA** usar K-Fold estándar (rompe dependencia temporal)
- Usar TimeSeriesSplit o Walk-Forward Validation
- Mantener orden cronológico en train/test

### 3. Features Exógenas
- Incorporar variables macroeconómicas puede mejorar significativamente el modelo
- Verificar correlación y causalidad de Granger
- Considerar desfases temporales (datos publicados con delay)

### 4. Gestión de Riesgo
- Implementar **intervalos de confianza** en predicciones
- Monitorear concept drift (cambios en distribución)
- Reentrenamiento periódico (semanal/mensual)

### 5. Seguridad
- **NUNCA** versionar credenciales en Git
- Usar variables de entorno o servicios de secretos
- Implementar rate limiting en API
- Validación de inputs en endpoints

## Métricas de Éxito

### Técnicas
- **MAPE < 2%**: Excelente
- **MAPE 2-5%**: Bueno
- **MAPE > 5%**: Requiere mejoras

### De Negocio
- Precisión en predicciones a 1 día: > 95%
- Precisión en predicciones a 7 días: > 85%
- Tiempo de inferencia: < 100ms

## Próximos Pasos Inmediatos

1. **Análisis Exploratorio de Datos (EDA)**
   - Estadísticas descriptivas
   - Detección de outliers
   - Análisis de estacionalidad
   - Prueba de estacionaridad (ADF test)

2. **Feature Engineering Avanzado**
   - Transformaciones (log, diferencias)
   - Features de Fourier para estacionalidad
   - Indicadores técnicos del mercado Forex

3. **Implementar 3 Modelos Comparativos**
   - Random Forest: Baseline ML robusto
   - XGBoost: Modelo avanzado de gradient boosting
   - AutoARIMA: Baseline estadístico clásico

4. **Comparación y Selección**
   - Tabla comparativa de métricas (MAE, RMSE, MAPE, R²)
   - Feature importance analysis (RF y XGBoost)
   - Seleccionar modelo ganador o implementar ensemble

## Recursos y Referencias

### APIs y Datos
- [Banco Central de Chile - API](https://si3.bcentral.cl/Siete/ES/Siete/Portada)
- [Yahoo Finance API](https://finance.yahoo.com/) (para cobre y otros commodities)

### Literatura
- "Forecasting: Principles and Practice" - Rob J Hyndman
- "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow"
- Papers sobre XGBoost para time series

### Herramientas
- [Optuna Documentation](https://optuna.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn Time Series](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)

## Notas del Desarrollador

### Estado Actual del Proyecto
- **Fase completada**: Módulos de adquisición y preprocesamiento de datos
- **Carpetas de datos**: Creadas pero vacías (pendiente primera descarga)
- **Entorno virtual**: Configurado en `/home/bastianberrios/proyectos/dolar_ml/venv/`
- **Control de versiones**: Git configurado con `.gitignore` (excluye `venv/`, `__pycache__/`, `.env`)

### Datos Descargados (Completado)
- **Serie**: F073.TCO.PRE.Z.D (USD/CLP observado diario)
- **Período**: 1995-10-03 a 2025-10-02 (30 años)
- **Observaciones totales**: 10,958 registros
- **Valores válidos**: 7,476 (68.2%)
- **Valores faltantes (ND)**: 3,482 (31.8% - fines de semana y feriados)

### Dataset Procesado con Features
- **Archivo**: data/processed/dolar_features.csv
- **Registros finales**: 3,993 (tras eliminar NaN)
- **Features totales**: 23 columnas
- **Reducción**: De 10,958 a 3,993 registros (63.5% de reducción por NaN en features lag)

### Próxima Acción Inmediata
1. Ejecutar notebooks para análisis exploratorio (01 y 02)
2. Implementar 3 modelos comparativos (Random Forest, XGBoost, ARIMA)
3. Crear notebook 03_feature_engineering.ipynb (documentación)
4. Crear notebook 04_modeling.ipynb con comparación de los 3 modelos
5. Seleccionar modelo ganador según métricas o implementar ensemble

---

**Última actualización**: Octubre 2025
**Estado del proyecto**: Fase 1-2 COMPLETADAS, Fase 3-5 pendientes
**Licencia**: Por definir

## Resumen de Avance

- [COMPLETADA] Fase 1: Adquisición de Datos - 10,958 registros descargados
- [COMPLETADA] Fase 2: Feature Engineering - 23 features implementadas, 3,993 registros procesados
- [PENDIENTE] Fase 3: Modelado Multi-Modelo (Random Forest + XGBoost + ARIMA)
- [PENDIENTE] Fase 4: Evaluación y Backtesting
- [PENDIENTE] Fase 5: Despliegue API

## Estrategia de Modelado

**Modelos a Implementar:**
1. **Random Forest** - Baseline ML, robusto y fácil de interpretar
2. **XGBoost** - Modelo avanzado, alto rendimiento
3. **AutoARIMA** - Baseline estadístico clásico

**Criterio de Selección:** Comparar MAE, RMSE, MAPE y R² en conjunto de test temporal
