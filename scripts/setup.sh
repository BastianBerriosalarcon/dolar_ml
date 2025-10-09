#!/bin/bash
# Script de setup completo para desarrollo

set -e  # Exit on error

echo "=== DolarCLP Predictor - Setup Completo ==="
echo ""

# 1. Verificar Python
echo "[1/8] Verificando Python 3.10+..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
if (( $(echo "$python_version < 3.10" | bc -l) )); then
    echo "[ERROR] Se requiere Python 3.10 o superior"
    exit 1
fi
echo "[OK] Python $python_version detectado"

# 2. Crear entorno virtual
echo ""
echo "[2/8] Creando entorno virtual..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "[OK] Entorno virtual creado"
else
    echo "[OK] Entorno virtual ya existe"
fi

# 3. Activar entorno
echo ""
echo "[3/8] Activando entorno virtual..."
source venv/bin/activate

# 4. Actualizar pip
echo ""
echo "[4/8] Actualizando pip..."
pip install --upgrade pip -q

# 5. Instalar dependencias
echo ""
echo "[5/8] Instalando dependencias..."
pip install -r requirements.txt -q
pip install -r requirements-dev.txt -q
echo "[OK] Dependencias instaladas"

# 6. Configurar pre-commit (opcional)
echo ""
echo "[6/8] Configurando pre-commit hooks (opcional)..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    echo "[OK] Pre-commit hooks configurados"
else
    echo "[WARNING] pre-commit no instalado (opcional)"
fi

# 7. Ejecutar tests
echo ""
echo "[7/8] Ejecutando tests..."
pytest tests/ -q --tb=short
echo "[OK] Tests ejecutados exitosamente"

# 8. Verificar estructura
echo ""
echo "[8/8] Verificando estructura del proyecto..."
required_dirs=("src" "tests" "data" "models" "notebooks")
for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "[OK] $dir/"
    else
        echo "[ERROR] $dir/ faltante"
    fi
done

echo ""
echo "=== Setup Completado Exitosamente! ==="
echo ""
echo "Proximos pasos:"
echo ""
echo "1. Configurar credenciales del BCCh:"
echo "   export BC_USER='tu_email@ejemplo.com'"
echo "   export BC_PASS='tu_password'"
echo ""
echo "2. Descargar datos:"
echo "   python download_data.py"
echo ""
echo "3. Generar features:"
echo "   python -m src.features.build_features"
echo ""
echo "4. Entrenar modelo:"
echo "   python -m src.models.train"
echo ""
echo "5. Ejecutar API:"
echo "   uvicorn src.api.app:app --reload"
echo ""
echo "6. Ver tests con coverage:"
echo "   pytest tests/ --cov=src --cov-report=html"
echo "   open htmlcov/index.html"
echo ""
