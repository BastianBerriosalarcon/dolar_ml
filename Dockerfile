# Imagen base con Python 3.12
FROM python:3.12-slim

# Metadata
LABEL maintainer="Bastian Berrios <bastianberrios.a@gmail.com>"
LABEL description="DolarCLP Predictor API - Sistema de predicción USD/CLP"

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar código fuente
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/ 2>/dev/null || true

# Crear directorios necesarios
RUN mkdir -p logs models data/raw data/processed

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Comando por defecto: iniciar API
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
