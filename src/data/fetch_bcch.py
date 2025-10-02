"""
Módulo para extracción de datos del Banco Central de Chile (BCCh).

Este módulo proporciona funciones para descargar series de tiempo económicas
desde la API REST del Banco Central de Chile.
"""

import os
import sys
import requests
from datetime import date, timedelta
from typing import Dict, Optional, Tuple


def get_bcch_credentials() -> Tuple[str, str]:
    """
    Obtiene las credenciales del BCCh desde variables de entorno.

    Returns:
        Tuple[str, str]: (usuario, contraseña)

    Raises:
        ValueError: Si las credenciales no están configuradas
    """
    bc_user = os.environ.get('BC_USER')
    bc_pass = os.environ.get('BC_PASS')

    if not bc_user or not bc_pass:
        raise ValueError(
            "Credenciales no configuradas. Define las variables de entorno:\n"
            "  export BC_USER='tu_email@ejemplo.com'\n"
            "  export BC_PASS='tu_contraseña'"
        )

    return bc_user, bc_pass


def fetch_dolar_observado(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    years_back: int = 30,
    verbose: bool = False
) -> Dict:
    """
    Descarga la serie de Dólar Observado del BCCh.

    Args:
        start_date: Fecha inicial en formato YYYY-MM-DD (opcional)
        end_date: Fecha final en formato YYYY-MM-DD (opcional)
        years_back: Años hacia atrás desde hoy si no se especifican fechas
        verbose: Mostrar información detallada

    Returns:
        Dict: Respuesta JSON de la API con las observaciones

    Raises:
        ValueError: Si hay error en las credenciales
        requests.RequestException: Si hay error en la petición HTTP
    """
    # Obtener credenciales
    bc_user, bc_pass = get_bcch_credentials()

    # Calcular fechas si no se proporcionan
    if not end_date:
        end_date = date.today().strftime("%Y-%m-%d")

    if not start_date:
        # Usar 365.25 para considerar años bisiestos
        start_date = (date.today() - timedelta(days=int(years_back*365.25))).strftime("%Y-%m-%d")

    # Configurar parámetros de la API
    api_url = 'https://si3.bcentral.cl/SieteRestWS/SieteRestWS.ashx'
    params = {
        'user': bc_user,
        'pass': bc_pass,
        'timeseries': 'F073.TCO.PRE.Z.D',  # Dólar observado diario
        'firstdate': start_date,
        'lastdate': end_date,
        'format': 'json'
    }

    if verbose:
        print(f"Consultando datos desde {start_date} hasta {end_date} con usuario {bc_user}...")

    # Realizar petición con timeout de 30 segundos
    response = requests.get(api_url, params=params, timeout=30)

    if response.status_code != 200:
        error_msg = f"Error HTTP {response.status_code}: {response.text}"
        raise requests.RequestException(error_msg)

    # Parsear respuesta JSON
    resp_json = response.json()

    # Validar respuesta del servicio
    codigo = resp_json.get('Codigo')
    descripcion = resp_json.get('Descripcion')

    if codigo is not None and codigo < 0:
        raise ValueError(f"Error del servicio (Codigo={codigo}): {descripcion}")

    # Extraer observaciones
    series = resp_json.get('Series', {})
    obs = series.get('Obs') or []

    if verbose:
        if not obs:
            print("La consulta no retornó observaciones (Obs vacío).")
        else:
            print(f"Datos descargados exitosamente. Observaciones recibidas: {len(obs)}")

    return resp_json


def fetch_series(
    series_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    verbose: bool = False
) -> Dict:
    """
    Descarga una serie genérica del BCCh.

    Args:
        series_code: Código de la serie (ej: 'F073.TCO.PRE.Z.D')
        start_date: Fecha inicial en formato YYYY-MM-DD
        end_date: Fecha final en formato YYYY-MM-DD
        verbose: Mostrar información detallada

    Returns:
        Dict: Respuesta JSON de la API
    """
    bc_user, bc_pass = get_bcch_credentials()

    if not end_date:
        end_date = date.today().strftime("%Y-%m-%d")

    if not start_date:
        # Usar 365.25 para considerar años bisiestos
        start_date = (date.today() - timedelta(days=int(30*365.25))).strftime("%Y-%m-%d")

    api_url = 'https://si3.bcentral.cl/SieteRestWS/SieteRestWS.ashx'
    params = {
        'user': bc_user,
        'pass': bc_pass,
        'timeseries': series_code,
        'firstdate': start_date,
        'lastdate': end_date,
        'format': 'json'
    }

    if verbose:
        print(f"Consultando serie {series_code} desde {start_date} hasta {end_date}...")

    response = requests.get(api_url, params=params, timeout=30)

    if response.status_code != 200:
        raise requests.RequestException(f"Error HTTP {response.status_code}: {response.text}")

    resp_json = response.json()
    codigo = resp_json.get('Codigo')
    descripcion = resp_json.get('Descripcion')

    if codigo is not None and codigo < 0:
        raise ValueError(f"Error del servicio (Codigo={codigo}): {descripcion}")

    return resp_json


if __name__ == "__main__":
    """Ejecutar como script para probar la extracción."""
    try:
        verbose = os.environ.get('DOLAR_VERBOSE', '0') == '1'
        data = fetch_dolar_observado(verbose=True)
        obs = data.get('Series', {}).get('Obs', [])
        print(f"\nExtracción exitosa: {len(obs)} observaciones")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
