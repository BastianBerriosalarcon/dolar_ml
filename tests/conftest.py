"""
Pytest configuration and shared fixtures.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_bcch_data():
    """
    Fixture con datos de ejemplo del BCCh.

    Returns:
        DataFrame con estructura similar a datos del Banco Central
    """
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    np.random.seed(42)

    # Simular valores realistas USD/CLP
    base_value = 800
    trend = np.linspace(0, 100, 100)
    noise = np.random.randn(100) * 10
    values = base_value + trend + noise

    # Simular algunos valores faltantes (fines de semana)
    status_codes = ['OK'] * 100
    for i, date in enumerate(dates):
        if date.weekday() >= 5:  # Sábado o domingo
            status_codes[i] = 'ND'

    return pd.DataFrame({
        'Valor': values,
        'statusCode': status_codes
    }, index=dates)


@pytest.fixture
def sample_features_data():
    """
    Fixture con datos de ejemplo con features engineering.

    Returns:
        DataFrame con features engineered
    """
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    np.random.seed(42)

    base_value = 800
    values = base_value + np.random.randn(100).cumsum() * 5

    df = pd.DataFrame({'Valor': values}, index=dates)

    # Agregar algunas features básicas
    df['Valor_lag_1'] = df['Valor'].shift(1)
    df['Valor_lag_7'] = df['Valor'].shift(7)
    df['MA_7'] = df['Valor'].rolling(window=7, min_periods=1).mean()
    df['MA_30'] = df['Valor'].rolling(window=30, min_periods=1).mean()

    return df.dropna()


@pytest.fixture
def sample_model_data():
    """
    Fixture con datos listos para modelado (sin NaN).

    Returns:
        Tuple de (X, y) para entrenamiento
    """
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    np.random.seed(42)

    # Features
    X = pd.DataFrame({
        'Valor_lag_1': np.random.randn(100) * 10 + 800,
        'Valor_lag_7': np.random.randn(100) * 10 + 800,
        'MA_7': np.random.randn(100) * 8 + 800,
        'MA_30': np.random.randn(100) * 6 + 800,
        'ROC_7': np.random.randn(100) * 2,
    }, index=dates)

    # Target (correlacionado con features)
    y = pd.Series(
        800 + 0.5 * (X['Valor_lag_1'] - 800) + np.random.randn(100) * 5,
        index=dates,
        name='Valor'
    )

    return X, y
