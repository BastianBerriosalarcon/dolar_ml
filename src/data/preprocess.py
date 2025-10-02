"""
Módulo para procesamiento y limpieza de datos del BCCh.

Proporciona funciones para transformar respuestas JSON de la API del BCCh
en DataFrames de Pandas listos para análisis.
"""

import os
import pandas as pd
from typing import Dict, Optional


def json_to_dataframe(resp_json: Dict, verbose: bool = False) -> Optional[pd.DataFrame]:
    """
    Convierte la respuesta JSON del BCCh a un DataFrame de Pandas.

    Args:
        resp_json: Respuesta JSON de la API del BCCh
        verbose: Mostrar información detallada del procesamiento

    Returns:
        pd.DataFrame: DataFrame con columnas [Fecha, Valor, statusCode]
                      o None si no hay observaciones

    Raises:
        ValueError: Si el formato JSON no es válido
    """
    # Extraer observaciones
    series = resp_json.get('Series', {})
    obs = series.get('Obs')

    if not obs:
        if verbose:
            print("No hay observaciones para procesar.")
        return None

    # Crear DataFrame
    df = pd.DataFrame(obs)

    if verbose:
        print(f"DataFrame creado con {len(df)} filas. Procesando...")

    # Mapear columnas a nombres amigables
    rename_map = {}
    if 'indexDateString' in df.columns:
        rename_map['indexDateString'] = 'Fecha'
    if 'index' in df.columns and 'Fecha' not in rename_map.values():
        rename_map['index'] = 'Fecha'
    if 'value' in df.columns:
        rename_map['value'] = 'Valor'

    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # Convertir 'Valor' a numérico (manejo de comas decimales y valores ND)
    if 'Valor' in df.columns:
        df['Valor'] = df['Valor'].str.replace(',', '.', regex=False)
        df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce')

    # Convertir 'Fecha' a datetime y establecer como índice
    if 'Fecha' in df.columns:
        df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d-%m-%Y', errors='coerce')
        df.set_index('Fecha', inplace=True)

    # Ordenar por fecha
    df.sort_index(inplace=True)

    return df


def print_summary(df: pd.DataFrame, verbose: bool = False) -> None:
    """
    Imprime un resumen estadístico del DataFrame.

    Args:
        df: DataFrame a resumir
        verbose: Mostrar información extendida
    """
    if df is None or df.empty:
        print("DataFrame vacío, no hay resumen disponible.")
        return

    total = len(df)
    non_null = int(df['Valor'].count()) if 'Valor' in df.columns else 0
    first = df.index.min().date() if total else None
    last = df.index.max().date() if total else None

    print(f"\nResumen: filas={total}, valores_no_nulos={non_null}, fecha_inicio={first}, fecha_fin={last}")

    # Estadísticas de statusCode si existe
    if 'statusCode' in df.columns:
        ok_count = int((df['statusCode'] == 'OK').sum())
        nd_count = int((df['statusCode'] == 'ND').sum())
        print(f"status OK={ok_count}, ND={nd_count}")

    # Información detallada si verbose
    if verbose:
        print("\nInformación del DataFrame (verbose):")
        df.info()
        print("\nEstadísticas descriptivas:")
        print(df.describe())

    # Mostrar primeras filas
    print("\nPrimeras 5 filas:")
    print(df.head())


def save_to_csv(df: pd.DataFrame, output_path: Optional[str] = None, verbose: bool = False) -> str:
    """
    Guarda el DataFrame en un archivo CSV.

    Args:
        df: DataFrame a guardar
        output_path: Ruta del archivo (opcional, por defecto: data/raw/dolar_bcch.csv)
        verbose: Mostrar información del guardado

    Returns:
        str: Ruta donde se guardó el archivo
    """
    if output_path is None:
        # Determinar el directorio raíz del proyecto
        # Buscar la raíz del proyecto (donde está src/)
        current_file = os.path.abspath(__file__)
        src_dir = os.path.dirname(os.path.dirname(current_file))  # src/data -> src
        project_root = os.path.dirname(src_dir)  # src -> proyecto raíz

        output_path = os.path.join(project_root, 'data', 'raw', 'dolar_bcch.csv')

    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Guardar CSV
    df.to_csv(output_path)

    if verbose:
        print(f"Archivo guardado en: {output_path}")

    return output_path


def process_and_save(
    resp_json: Dict,
    output_path: Optional[str] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Pipeline completo: JSON -> DataFrame -> CSV -> Resumen.

    Args:
        resp_json: Respuesta JSON de la API
        output_path: Ruta de salida (opcional)
        verbose: Mostrar información detallada

    Returns:
        pd.DataFrame: DataFrame procesado
    """
    # Convertir a DataFrame
    df = json_to_dataframe(resp_json, verbose=verbose)

    if df is None or df.empty:
        print("No se pudo crear el DataFrame.")
        return None

    # Guardar a CSV
    saved_path = save_to_csv(df, output_path, verbose=verbose)
    print(f"\nArchivo guardado en: {saved_path}")

    # Imprimir resumen
    print_summary(df, verbose=verbose)

    return df


if __name__ == "__main__":
    """Ejecutar como script para probar el procesamiento."""
    import sys
    from fetch_bcch import fetch_dolar_observado

    try:
        verbose = os.environ.get('DOLAR_VERBOSE', '0') == '1'
        print("Descargando datos...")
        data = fetch_dolar_observado(verbose=verbose)

        print("\nProcesando datos...")
        df = process_and_save(data, verbose=verbose)

        if df is not None:
            print(f"\nProcesamiento exitoso: {len(df)} registros guardados")
        else:
            print("\nNo se pudo procesar los datos")
            sys.exit(1)

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
