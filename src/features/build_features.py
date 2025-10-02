"""
Módulo para construir features (características) para modelos de ML.

Este módulo proporciona funciones para crear variables derivadas
a partir de datos de series de tiempo financieras.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def add_lag_features(
    df: pd.DataFrame,
    column: str = 'Valor',
    lags: List[int] = [1, 7, 30]
) -> pd.DataFrame:
    """
    Agrega variables lag (valores históricos desplazados).

    Args:
        df: DataFrame con serie temporal
        column: Nombre de la columna a procesar
        lags: Lista de períodos de desplazamiento

    Returns:
        DataFrame con columnas lag agregadas
    """
    df_copy = df.copy()

    for lag in lags:
        df_copy[f'{column}_lag_{lag}'] = df_copy[column].shift(lag)

    return df_copy


def add_moving_averages(
    df: pd.DataFrame,
    column: str = 'Valor',
    windows: List[int] = [7, 30, 90]
) -> pd.DataFrame:
    """
    Agrega promedios móviles (Moving Averages).

    Args:
        df: DataFrame con serie temporal
        column: Nombre de la columna a procesar
        windows: Lista de ventanas para promedios móviles

    Returns:
        DataFrame con columnas MA agregadas
    """
    df_copy = df.copy()

    for window in windows:
        df_copy[f'MA_{window}'] = df_copy[column].rolling(window=window, min_periods=1).mean()

    return df_copy


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega características temporales (día de semana, mes, trimestre).

    Args:
        df: DataFrame con índice de tipo DatetimeIndex

    Returns:
        DataFrame con columnas temporales agregadas
    """
    df_copy = df.copy()

    # Asegurar que el índice es DatetimeIndex
    if not isinstance(df_copy.index, pd.DatetimeIndex):
        raise ValueError("El índice debe ser de tipo DatetimeIndex")

    df_copy['day_of_week'] = df_copy.index.dayofweek
    df_copy['day_of_month'] = df_copy.index.day
    df_copy['month'] = df_copy.index.month
    df_copy['quarter'] = df_copy.index.quarter
    df_copy['year'] = df_copy.index.year
    df_copy['is_month_start'] = df_copy.index.is_month_start.astype(int)
    df_copy['is_month_end'] = df_copy.index.is_month_end.astype(int)

    return df_copy


def add_technical_indicators(
    df: pd.DataFrame,
    column: str = 'Valor'
) -> pd.DataFrame:
    """
    Agrega indicadores técnicos (ROC, Momentum).

    Args:
        df: DataFrame con serie temporal
        column: Nombre de la columna a procesar

    Returns:
        DataFrame con indicadores técnicos agregados
    """
    df_copy = df.copy()

    # Rate of Change (ROC) - Cambio porcentual
    df_copy['ROC_7'] = df_copy[column].pct_change(periods=7, fill_method=None) * 100
    df_copy['ROC_30'] = df_copy[column].pct_change(periods=30, fill_method=None) * 100

    # Momentum
    df_copy['momentum_7'] = df_copy[column] - df_copy[column].shift(7)
    df_copy['momentum_30'] = df_copy[column] - df_copy[column].shift(30)

    # Retorno diario
    df_copy['daily_return'] = df_copy[column].pct_change(fill_method=None) * 100

    return df_copy


def add_volatility_features(
    df: pd.DataFrame,
    column: str = 'Valor',
    windows: List[int] = [7, 30]
) -> pd.DataFrame:
    """
    Agrega características de volatilidad (desviación estándar móvil).

    Args:
        df: DataFrame con serie temporal
        column: Nombre de la columna a procesar
        windows: Lista de ventanas para calcular volatilidad

    Returns:
        DataFrame con columnas de volatilidad agregadas
    """
    df_copy = df.copy()

    # Calcular retornos diarios si no existen
    if 'daily_return' not in df_copy.columns:
        df_copy['daily_return'] = df_copy[column].pct_change() * 100

    # Volatilidad como desviación estándar de retornos
    for window in windows:
        df_copy[f'volatility_{window}'] = df_copy['daily_return'].rolling(window=window, min_periods=1).std()

    # Rango (High-Low) como proxy de volatilidad intraday
    # Usamos diferencia entre max y min en ventana
    for window in windows:
        df_copy[f'range_{window}'] = (
            df_copy[column].rolling(window=window, min_periods=1).max() -
            df_copy[column].rolling(window=window, min_periods=1).min()
        )

    return df_copy


def build_all_features(
    df: pd.DataFrame,
    column: str = 'Valor',
    lag_periods: List[int] = [1, 7, 30],
    ma_windows: List[int] = [7, 30, 90],
    vol_windows: List[int] = [7, 30]
) -> pd.DataFrame:
    """
    Pipeline completo: agrega todas las features al DataFrame.

    Args:
        df: DataFrame con serie temporal
        column: Nombre de la columna a procesar
        lag_periods: Períodos de lag
        ma_windows: Ventanas para moving averages
        vol_windows: Ventanas para volatilidad

    Returns:
        DataFrame con todas las features agregadas
    """
    df_features = df.copy()

    print("Construyendo features...")

    # 1. Lags
    print(f"  - Agregando lags: {lag_periods}")
    df_features = add_lag_features(df_features, column, lag_periods)

    # 2. Moving Averages
    print(f"  - Agregando moving averages: {ma_windows}")
    df_features = add_moving_averages(df_features, column, ma_windows)

    # 3. Características temporales
    print("  - Agregando características temporales")
    df_features = add_temporal_features(df_features)

    # 4. Indicadores técnicos
    print("  - Agregando indicadores técnicos (ROC, Momentum)")
    df_features = add_technical_indicators(df_features, column)

    # 5. Volatilidad
    print(f"  - Agregando features de volatilidad: {vol_windows}")
    df_features = add_volatility_features(df_features, column, vol_windows)

    print(f"Features construidas: {len(df_features.columns)} columnas totales")

    return df_features


def prepare_for_ml(
    df: pd.DataFrame,
    target_column: str = 'Valor',
    drop_na: bool = True,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Prepara el DataFrame para machine learning.

    Elimina columnas innecesarias y maneja valores faltantes.

    Args:
        df: DataFrame con features
        target_column: Nombre de la columna objetivo
        drop_na: Si True, elimina filas con NaN
        verbose: Mostrar información del proceso

    Returns:
        DataFrame listo para ML
    """
    df_ml = df.copy()

    # Eliminar columna statusCode si existe
    if 'statusCode' in df_ml.columns:
        df_ml = df_ml.drop(columns=['statusCode'])

    # Información antes de limpiar
    if verbose:
        print(f"\nRegistros antes de limpiar: {len(df_ml)}")
        print(f"Columnas: {len(df_ml.columns)}")
        print(f"\nValores faltantes por columna:")
        print(df_ml.isnull().sum())

    # Eliminar filas con NaN si se solicita
    if drop_na:
        df_ml = df_ml.dropna()
        if verbose:
            print(f"\nRegistros después de eliminar NaN: {len(df_ml)}")

    if verbose:
        print(f"\nDataFrame final: {df_ml.shape}")
        print(f"Columnas: {list(df_ml.columns)}")

    return df_ml


if __name__ == "__main__":
    """Script de prueba para feature engineering."""
    import sys
    import os

    # Cargar datos
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'raw', 'dolar_bcch.csv')

    if not os.path.exists(data_path):
        print(f"ERROR: No se encontró el archivo {data_path}")
        print("Ejecuta primero download_data.py para descargar los datos")
        sys.exit(1)

    print(f"Cargando datos desde {data_path}...")
    df = pd.read_csv(data_path, index_col='Fecha', parse_dates=True)
    print(f"Datos cargados: {len(df)} registros\n")

    # Construir features
    df_features = build_all_features(df)

    # Preparar para ML
    print("\nPreparando datos para ML...")
    df_ml = prepare_for_ml(df_features, verbose=True)

    # Guardar dataset procesado
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'processed', 'dolar_features.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_ml.to_csv(output_path)
    print(f"\nDataset con features guardado en: {output_path}")
    print(f"Registros finales: {len(df_ml)}")
    print(f"Features totales: {len(df_ml.columns)}")

    # Mostrar primeras filas
    print("\nPrimeras 5 filas del dataset:")
    print(df_ml.head())
