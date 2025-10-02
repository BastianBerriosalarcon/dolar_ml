# DolarCLP-Predictor

Sistema de predicción del tipo de cambio USD/CLP usando Machine Learning.

## Descripción

Proyecto de Machine Learning para predecir el tipo de cambio del dólar estadounidense (USD) frente al peso chileno (CLP) utilizando análisis de series de tiempo y XGBoost.

### Características

- Descarga automática de datos históricos del Banco Central de Chile (30 años)
- Feature engineering completo con 23 características
- Análisis exploratorio de datos (EDA)
- Modelo predictivo basado en XGBoost (en desarrollo)

## Estado del Proyecto

- [COMPLETADA] Fase 1: Adquisición de Datos - 10,958 registros
- [COMPLETADA] Fase 2: Feature Engineering - 23 features
- [PENDIENTE] Fase 3: Modelado XGBoost
- [PENDIENTE] Fase 4: Evaluación y Backtesting
- [PENDIENTE] Fase 5: Despliegue API

## Estructura del Proyecto

```
dolar_ml/
├── data/
│   ├── raw/              # Datos crudos del BCCh
│   └── processed/        # Datos procesados con features
├── notebooks/            # Jupyter notebooks para análisis
├── src/
│   ├── data/            # Módulos de adquisición de datos
│   └── features/        # Feature engineering
├── download_data.py     # Script de descarga
└── requirements.txt     # Dependencias
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
jupyter notebook notebooks/02_eda.ipynb
```

## Dataset

- **Fuente**: Banco Central de Chile
- **Serie**: F073.TCO.PRE.Z.D (Dólar observado diario)
- **Período**: 1995-2025 (30 años)
- **Registros totales**: 10,958
- **Registros procesados**: 3,993 (con features completas)

## Features Implementadas

- **Lags**: t-1, t-7, t-30
- **Moving Averages**: 7, 30, 90 días
- **Temporales**: día, mes, trimestre, año
- **Técnicas**: ROC, Momentum, retornos diarios
- **Volatilidad**: desviación estándar móvil, rangos

## Tecnologías

- Python 3.12
- Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn
- Jupyter Notebook

## Contribuir

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## Licencia

Por definir

## Autor

Bastián Berríos - bastianberrios.a@gmail.com

## Agradecimientos

- Banco Central de Chile por proporcionar la API de datos
