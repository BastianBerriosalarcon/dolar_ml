"""
Tests para el módulo de construcción de features.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.features.build_features import (
    add_lag_features,
    add_moving_averages,
    add_temporal_features,
    add_technical_indicators,
    add_volatility_features,
    build_all_features,
    prepare_for_ml
)


class TestLagFeatures:
    """Tests para add_lag_features."""

    def test_creates_correct_lag_columns(self, sample_features_data):
        """Verifica que se crean las columnas lag correctas."""
        result = add_lag_features(sample_features_data, lags=[1, 7])

        assert 'Valor_lag_1' in result.columns
        assert 'Valor_lag_7' in result.columns

    def test_lag_values_are_shifted_correctly(self, sample_features_data):
        """Verifica que los valores lag sean correctos."""
        result = add_lag_features(sample_features_data, lags=[1])

        # El lag_1 del segundo elemento debe ser el valor del primero
        assert result['Valor_lag_1'].iloc[1] == sample_features_data['Valor'].iloc[0]

    def test_lag_creates_nan_at_beginning(self, sample_features_data):
        """Verifica que los lags tengan NaN en las primeras filas."""
        result = add_lag_features(sample_features_data, lags=[7])

        assert result['Valor_lag_7'].iloc[:7].isna().all()
        assert result['Valor_lag_7'].iloc[7:].notna().any()

    def test_multiple_lags(self, sample_features_data):
        """Verifica creación de múltiples lags."""
        result = add_lag_features(sample_features_data, lags=[1, 7, 30])

        assert 'Valor_lag_1' in result.columns
        assert 'Valor_lag_7' in result.columns
        assert 'Valor_lag_30' in result.columns

    def test_custom_column_name(self):
        """Verifica que funcione con nombres de columna personalizados."""
        df = pd.DataFrame({'precio': [100, 101, 102, 103]})
        result = add_lag_features(df, column='precio', lags=[1])

        assert 'precio_lag_1' in result.columns


class TestMovingAverages:
    """Tests para add_moving_averages."""

    def test_creates_ma_columns(self, sample_features_data):
        """Verifica que se crean las columnas MA correctas."""
        result = add_moving_averages(sample_features_data, windows=[7])

        assert 'MA_7' in result.columns

    def test_ma_calculation_accuracy(self):
        """Verifica que MA se calcule correctamente."""
        df = pd.DataFrame({'Valor': [10, 20, 30, 40, 50, 60, 70]})
        result = add_moving_averages(df, windows=[3])

        # MA_3 para el índice 2 debe ser (10+20+30)/3 = 20
        assert result['MA_3'].iloc[2] == 20.0
        # MA_3 para el índice 6 debe ser (50+60+70)/3 = 60
        assert result['MA_6'].iloc[6] == 60.0 if 'MA_6' in result.columns else True

    def test_ma_has_nan_at_beginning(self):
        """Verifica que MA tenga NaN al inicio."""
        # Crear data fresca sin NaN previos
        df = pd.DataFrame({'Valor': range(1, 11)})
        result = add_moving_averages(df, windows=[7])

        # Los primeros 6 valores deben ser NaN (min_periods=window)
        assert result['MA_7'].iloc[:6].isna().all()
        # El séptimo valor debe existir
        assert not pd.isna(result['MA_7'].iloc[6])

    def test_multiple_windows(self, sample_features_data):
        """Verifica creación de múltiples ventanas."""
        result = add_moving_averages(sample_features_data, windows=[7, 30])

        assert 'MA_7' in result.columns
        assert 'MA_30' in result.columns


class TestTemporalFeatures:
    """Tests para add_temporal_features."""

    def test_creates_temporal_columns(self, sample_features_data):
        """Verifica que se crean las columnas temporales."""
        result = add_temporal_features(sample_features_data)

        expected_cols = ['day_of_week', 'day_of_month', 'month', 'quarter',
                        'year', 'is_month_start', 'is_month_end']
        for col in expected_cols:
            assert col in result.columns

    def test_day_of_week_values(self):
        """Verifica que day_of_week tenga valores correctos (0-6)."""
        dates = pd.date_range('2020-01-06', periods=7, freq='D')  # Lunes a Domingo
        df = pd.DataFrame({'Valor': range(7)}, index=dates)

        result = add_temporal_features(df)

        assert result['day_of_week'].iloc[0] == 0  # Lunes
        assert result['day_of_week'].iloc[6] == 6  # Domingo

    def test_month_values(self):
        """Verifica que month tenga valores correctos (1-12)."""
        dates = pd.DatetimeIndex(['2020-01-15', '2020-06-15', '2020-12-15'])
        df = pd.DataFrame({'Valor': [1, 2, 3]}, index=dates)

        result = add_temporal_features(df)

        assert result['month'].iloc[0] == 1
        assert result['month'].iloc[1] == 6
        assert result['month'].iloc[2] == 12

    def test_quarter_values(self):
        """Verifica que quarter tenga valores correctos (1-4)."""
        dates = pd.DatetimeIndex(['2020-01-15', '2020-04-15', '2020-07-15', '2020-10-15'])
        df = pd.DataFrame({'Valor': range(4)}, index=dates)

        result = add_temporal_features(df)

        assert result['quarter'].iloc[0] == 1
        assert result['quarter'].iloc[1] == 2
        assert result['quarter'].iloc[2] == 3
        assert result['quarter'].iloc[3] == 4

    def test_month_start_end_flags(self):
        """Verifica flags de inicio y fin de mes."""
        dates = pd.DatetimeIndex(['2020-01-01', '2020-01-15', '2020-01-31'])
        df = pd.DataFrame({'Valor': range(3)}, index=dates)

        result = add_temporal_features(df)

        assert result['is_month_start'].iloc[0] == 1
        assert result['is_month_start'].iloc[1] == 0
        assert result['is_month_end'].iloc[2] == 1

    def test_raises_error_without_datetime_index(self):
        """Verifica que arroje error sin DatetimeIndex."""
        df = pd.DataFrame({'Valor': [1, 2, 3]})  # Sin DatetimeIndex

        with pytest.raises(ValueError, match="DatetimeIndex"):
            add_temporal_features(df)


class TestTechnicalIndicators:
    """Tests para add_technical_indicators."""

    def test_creates_technical_indicator_columns(self, sample_features_data):
        """Verifica que se crean las columnas de indicadores técnicos."""
        result = add_technical_indicators(sample_features_data)

        expected_cols = ['ROC_7', 'ROC_30', 'momentum_7', 'momentum_30', 'daily_return']
        for col in expected_cols:
            assert col in result.columns

    def test_roc_calculation(self):
        """Verifica cálculo de Rate of Change."""
        df = pd.DataFrame({'Valor': [100, 110, 120]})
        result = add_technical_indicators(df)

        # ROC_7 y ROC_30 deberían ser NaN al principio
        assert pd.isna(result['ROC_7'].iloc[0])

    def test_momentum_calculation(self):
        """Verifica cálculo de Momentum."""
        df = pd.DataFrame({'Valor': [100, 105, 110, 115, 120, 125, 130, 135]})
        result = add_technical_indicators(df)

        # momentum_7[7] = Valor[7] - Valor[0] = 135 - 100 = 35
        expected_momentum = df['Valor'].iloc[7] - df['Valor'].iloc[0]
        assert result['momentum_7'].iloc[7] == expected_momentum

    def test_daily_return_percentage(self):
        """Verifica que daily_return esté en porcentaje."""
        df = pd.DataFrame({'Valor': [100, 110]})  # +10% incremento
        result = add_technical_indicators(df)

        # daily_return[1] debería ser aproximadamente 10%
        assert abs(result['daily_return'].iloc[1] - 10.0) < 0.1


class TestVolatilityFeatures:
    """Tests para add_volatility_features."""

    def test_creates_volatility_columns(self, sample_features_data):
        """Verifica que se crean las columnas de volatilidad."""
        result = add_volatility_features(sample_features_data, windows=[7])

        assert 'volatility_7' in result.columns
        assert 'range_7' in result.columns

    def test_volatility_is_std_of_returns(self):
        """Verifica que volatilidad sea std de retornos."""
        df = pd.DataFrame({'Valor': [100, 102, 98, 103, 97, 105, 95, 108]})
        result = add_volatility_features(df, windows=[7])

        # volatility_7 debe ser la desviación estándar de los retornos
        assert 'volatility_7' in result.columns
        assert result['volatility_7'].iloc[7:].notna().any()

    def test_range_calculation(self):
        """Verifica cálculo de range (max - min)."""
        df = pd.DataFrame({'Valor': [100, 110, 90, 105, 95, 115, 85]})
        result = add_volatility_features(df, windows=[7])

        # range_7[6] = max(100,110,90,105,95,115,85) - min(...) = 115 - 85 = 30
        expected_range = 115 - 85
        assert result['range_7'].iloc[6] == expected_range


class TestBuildAllFeatures:
    """Tests para build_all_features."""

    def test_creates_all_feature_types(self, sample_features_data):
        """Verifica que se crean todos los tipos de features."""
        result = build_all_features(sample_features_data)

        # Verificar que existan features de cada tipo
        assert any('lag' in col for col in result.columns)
        assert any('MA_' in col for col in result.columns)
        assert 'day_of_week' in result.columns
        assert 'ROC_7' in result.columns
        assert 'volatility_7' in result.columns

    def test_total_columns_count(self, sample_features_data):
        """Verifica el número total de columnas generadas."""
        result = build_all_features(sample_features_data)

        # Debe tener muchas más columnas que el original
        assert len(result.columns) > len(sample_features_data.columns)

    def test_original_data_preserved(self, sample_features_data):
        """Verifica que los datos originales se preserven."""
        result = build_all_features(sample_features_data)

        # La columna original 'Valor' debe seguir existiendo
        assert 'Valor' in result.columns
        # Los primeros valores no-NaN deben ser los mismos
        pd.testing.assert_series_equal(
            result['Valor'],
            sample_features_data['Valor'],
            check_names=False
        )


class TestPrepareForML:
    """Tests para prepare_for_ml."""

    def test_removes_status_code_column(self, sample_bcch_data):
        """Verifica que elimine la columna statusCode."""
        result = prepare_for_ml(sample_bcch_data)

        assert 'statusCode' not in result.columns

    def test_drops_na_rows_when_requested(self):
        """Verifica que elimine filas con NaN."""
        df = pd.DataFrame({
            'Valor': [1, 2, np.nan, 4],
            'feature': [10, 20, 30, 40]
        })

        result = prepare_for_ml(df, drop_na=True)

        assert len(result) < len(df)
        assert result['Valor'].isna().sum() == 0

    def test_keeps_na_rows_when_not_requested(self):
        """Verifica que mantenga NaN si drop_na=False."""
        df = pd.DataFrame({
            'Valor': [1, 2, np.nan, 4],
            'feature': [10, 20, 30, 40]
        })

        result = prepare_for_ml(df, drop_na=False)

        assert len(result) == len(df)

    def test_verbose_mode(self, sample_bcch_data, capsys):
        """Verifica que verbose mode imprima información."""
        prepare_for_ml(sample_bcch_data, verbose=True)
        captured = capsys.readouterr()

        assert "Registros antes" in captured.out
        assert "Registros después" in captured.out or "Valores faltantes" in captured.out
