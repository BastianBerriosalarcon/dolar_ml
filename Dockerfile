# Multi-stage build para optimizar tamaño de imagen

# Stage 1: Builder
FROM python:3.12-slim as builder

WORKDIR /app

# Instalar dependencias de build
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.12-slim

WORKDIR /app

# Crear usuario no-root para seguridad
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copiar dependencias instaladas desde builder
COPY --from=builder /root/.local /home/appuser/.local

# Copiar código fuente
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser models/ ./models/

# Configurar PATH
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Cambiar a usuario no-root
USER appuser

# Exponer puerto
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Comando por defecto
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
