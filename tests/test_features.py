"""
Tests para feature engineering
"""
import pytest
import pandas as pd
import numpy as np
from src.features.build_features import (
    add_lag_features,
    add_moving_averages,
    add_temporal_features,
    add_technical_indicators,
    build_all_features
)


@pytest.fixture
def sample_df():
    """Fixture con DataFrame de ejemplo"""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    data = {
        'Fecha': dates,
        'Valor': np.random.uniform(900, 950, 100)
    }
    df = pd.DataFrame(data)
    df.set_index('Fecha', inplace=True)
    return df


def test_add_lag_features(sample_df):
    """Test de features lag"""
    result = add_lag_features(sample_df.copy())
    
    assert 'Valor_lag_1' in result.columns
    assert 'Valor_lag_7' in result.columns
    assert 'Valor_lag_30' in result.columns
    
    assert result['Valor_lag_1'].isna().sum() >= 1
    assert result['Valor_lag_7'].isna().sum() >= 7


def test_add_moving_averages(sample_df):
    """Test de moving averages"""
    result = add_moving_averages(sample_df.copy())
    
    assert 'MA_7' in result.columns
    assert 'MA_30' in result.columns
    assert 'MA_90' in result.columns
    
    assert result['MA_7'].notna().sum() > 0


def test_add_temporal_features(sample_df):
    """Test de features temporales"""
    result = add_temporal_features(sample_df.copy())
    
    assert 'day_of_week' in result.columns
    assert 'month' in result.columns
    assert 'quarter' in result.columns
    assert 'year' in result.columns
    
    assert result['month'].min() >= 1
    assert result['month'].max() <= 12


def test_add_technical_indicators(sample_df):
    """Test de indicadores técnicos"""
    result = add_technical_indicators(sample_df.copy())
    
    assert 'ROC_7' in result.columns
    assert 'momentum_7' in result.columns
    assert 'daily_return' in result.columns


def test_build_all_features(sample_df):
    """Test de construcción completa de features"""
    result = build_all_features(sample_df.copy())
    
    assert len(result.columns) > len(sample_df.columns)
    
    expected_features = ['Valor_lag_1', 'MA_7', 'day_of_week', 'ROC_7']
    for feature in expected_features:
        assert feature in result.columns
    
    assert result['Valor'].notna().all()
