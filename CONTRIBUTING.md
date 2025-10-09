# Guía de Contribución

Gracias por tu interés en contribuir a DolarCLP-Predictor!

## Cómo Contribuir

### 1. Fork y Clone

```bash
# Fork el repositorio en GitHub, luego:
git clone https://github.com/tu-usuario/dolar_ml.git
cd dolar_ml
```

### 2. Configurar Entorno de Desarrollo

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias de desarrollo
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Instalar pre-commit hooks
pip install pre-commit
pre-commit install
```

### 3. Crear una Rama

```bash
git checkout -b feature/mi-nueva-feature
# o
git checkout -b fix/correccion-de-bug
```

### 4. Hacer Cambios

- Escribe código limpio y bien documentado
- Agrega tests para nuevo código
- Asegúrate de que todos los tests pasen
- Sigue las convenciones de código del proyecto

### 5. Ejecutar Tests

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Con cobertura
pytest tests/ --cov=src --cov-report=html

# La cobertura debe mantenerse > 75%
open htmlcov/index.html
```

### 6. Formatear Código

```bash
# Black (formatter)
black src/ tests/

# isort (import sorting)
isort src/ tests/

# flake8 (linter)
flake8 src/ tests/

# mypy (type checking)
mypy src/ --ignore-missing-imports
```

### 7. Commit y Push

```bash
# Commits descriptivos
git add .
git commit -m "feat: agregar nueva característica X"
git push origin feature/mi-nueva-feature
```

### 8. Pull Request

1. Ve a GitHub y crea un Pull Request
2. Describe los cambios realizados
3. Vincula issues relacionados
4. Espera review y feedback

## Estándares de Código

### Estilo Python

- Seguir **PEP 8**
- Líneas máximo 100 caracteres
- Usar **type hints** en firmas de funciones
- Documentar con **docstrings** (Google style)

### Ejemplo de Docstring

```python
def calcular_metrica(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula métrica de evaluación.

    Args:
        y_true: Valores reales
        y_pred: Valores predichos

    Returns:
        float: Valor de la métrica

    Raises:
        ValueError: Si los arrays tienen diferente tamaño
    """
    pass
```

### Commits

Usar **Conventional Commits**:

- `feat:` Nueva funcionalidad
- `fix:` Corrección de bug
- `docs:` Cambios en documentación
- `test:` Agregar o modificar tests
- `refactor:` Refactorización de código
- `style:` Cambios de formato (sin afectar lógica)
- `chore:` Tareas de mantenimiento

Ejemplos:
```
feat: agregar endpoint de predicción batch
fix: corregir cálculo de volatilidad en build_features
docs: actualizar README con ejemplos de API
test: agregar tests para modelo de predicción
```

## Tests

### Requisitos

- Todo nuevo código debe tener tests
- Cobertura mínima: 75%
- Tests deben ser independientes y reproducibles
- Usar fixtures para datos de prueba

### Estructura

```
tests/
├── conftest.py           # Fixtures compartidos
├── test_data/
│   └── test_*.py        # Tests para src/data/
├── test_features/
│   └── test_*.py        # Tests para src/features/
└── test_models/
    └── test_*.py        # Tests para src/models/
```

### Ejemplo de Test

```python
def test_add_lag_features(sample_data):
    """Verifica que se agreguen features lag correctamente."""
    result = add_lag_features(sample_data, lags=[1, 7])

    assert 'Valor_lag_1' in result.columns
    assert 'Valor_lag_7' in result.columns
    assert result['Valor_lag_1'].iloc[1] == sample_data['Valor'].iloc[0]
```

## Reportar Issues

### Bugs

Incluir:
- Descripción clara del problema
- Pasos para reproducir
- Comportamiento esperado vs actual
- Versión de Python y dependencias
- Stack trace si aplica

### Feature Requests

Incluir:
- Descripción de la funcionalidad
- Caso de uso
- Beneficio esperado
- Posible implementación (opcional)

## Code Review

### Para Reviewers

- Revisar lógica y corrección
- Verificar cobertura de tests
- Validar estilo y documentación
- Sugerir mejoras constructivamente

### Para Contributors

- Responder feedback prontamente
- Hacer cambios solicitados
- Discutir desacuerdos respetuosamente
- Actualizar PR según comentarios

## Preguntas

Si tienes dudas:
- Abre un issue con label `question`
- Contacta al maintainer: bastianberrios.a@gmail.com

Gracias por contribuir!
