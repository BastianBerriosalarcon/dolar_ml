#!/usr/bin/env python3
"""
Script para descargar datos históricos del BCCh y guardarlos en CSV.
"""

import os
import sys

# Asegurar que src esté en el path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.fetch_bcch import fetch_dolar_observado
from src.data.preprocess import process_and_save

if __name__ == "__main__":
    try:
        print("=" * 70)
        print("DESCARGA DE DATOS HISTÓRICOS USD/CLP - BANCO CENTRAL DE CHILE")
        print("=" * 70)

        # Verificar credenciales
        bc_user = os.environ.get('BC_USER')
        bc_pass = os.environ.get('BC_PASS')

        if not bc_user or not bc_pass:
            print("\nERROR: Credenciales no configuradas")
            print("\nConfigura las variables de entorno:")
            print("  export BC_USER='tu_email@ejemplo.com'")
            print("  export BC_PASS='tu_contraseña'")
            sys.exit(1)

        print(f"\nCredenciales configuradas para: {bc_user}")

        # Descargar datos
        print("\nDescargando datos históricos (30 años)...")
        print("Serie: F073.TCO.PRE.Z.D (Dólar observado diario)\n")

        data = fetch_dolar_observado(years_back=30, verbose=True)

        # Procesar y guardar
        print("\nProcesando y guardando datos...")
        df = process_and_save(data, verbose=True)

        if df is not None and not df.empty:
            print("\n" + "=" * 70)
            print("DESCARGA COMPLETADA EXITOSAMENTE")
            print("=" * 70)
            print(f"\nArchivo guardado en: data/raw/dolar_bcch.csv")
            print(f"Total de registros: {len(df):,}")
            print(f"Rango: {df.index.min().date()} a {df.index.max().date()}")
            print(f"Valores válidos: {df['Valor'].notna().sum():,}")
            print(f"Valores faltantes: {df['Valor'].isna().sum():,}")
            print("\nSiguiente paso: Ejecutar notebooks/01_data_acquisition.ipynb")
        else:
            print("\nERROR: No se pudo procesar los datos")
            sys.exit(1)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
