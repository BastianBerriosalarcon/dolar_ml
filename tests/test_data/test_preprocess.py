"""
Tests para el módulo de preprocessing de datos.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os

from src.data.preprocess import (
    json_to_dataframe,
    save_to_csv,
    process_and_save
)


class TestJsonToDataframe:
    """Tests para la función json_to_dataframe."""

    def test_valid_json_conversion(self):
        """Verifica conversión exitosa de JSON válido."""
        json_data = {
            'Series': {
                'Obs': [
                    {'indexDateString': '01-01-2020', 'value': '800,50', 'statusCode': 'OK'},
                    {'indexDateString': '02-01-2020', 'value': '801,25', 'statusCode': 'OK'},
                    {'indexDateString': '03-01-2020', 'value': 'ND', 'statusCode': 'ND'},
                ]
            }
        }

        df = json_to_dataframe(json_data)

        assert df is not None
        assert len(df) == 3
        assert 'Valor' in df.columns
        assert 'statusCode' in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_valor_conversion(self):
        """Verifica conversión correcta de valores con coma a float."""
        json_data = {
            'Series': {
                'Obs': [
                    {'indexDateString': '01-01-2020', 'value': '800,50', 'statusCode': 'OK'},
                    {'indexDateString': '02-01-2020', 'value': '801,25', 'statusCode': 'OK'},
                ]
            }
        }

        df = json_to_dataframe(json_data)

        assert df['Valor'].iloc[0] == 800.50
        assert df['Valor'].iloc[1] == 801.25
        assert df['Valor'].dtype == np.float64

    def test_nd_values_become_nan(self):
        """Verifica que valores 'ND' se conviertan en NaN."""
        json_data = {
            'Series': {
                'Obs': [
                    {'indexDateString': '01-01-2020', 'value': '800,50', 'statusCode': 'OK'},
                    {'indexDateString': '02-01-2020', 'value': 'ND', 'statusCode': 'ND'},
                ]
            }
        }

        df = json_to_dataframe(json_data)

        assert not pd.isna(df['Valor'].iloc[0])
        assert pd.isna(df['Valor'].iloc[1])

    def test_date_parsing(self):
        """Verifica parseo correcto de fechas."""
        json_data = {
            'Series': {
                'Obs': [
                    {'indexDateString': '15-03-2020', 'value': '800,50', 'statusCode': 'OK'},
                ]
            }
        }

        df = json_to_dataframe(json_data)

        assert df.index[0] == pd.Timestamp('2020-03-15')

    def test_empty_observations(self):
        """Verifica manejo de JSON sin observaciones."""
        json_data = {'Series': {}}

        df = json_to_dataframe(json_data)

        assert df is None

    def test_sorting_by_date(self):
        """Verifica que el DataFrame se ordene por fecha."""
        json_data = {
            'Series': {
                'Obs': [
                    {'indexDateString': '03-01-2020', 'value': '802,00', 'statusCode': 'OK'},
                    {'indexDateString': '01-01-2020', 'value': '800,00', 'statusCode': 'OK'},
                    {'indexDateString': '02-01-2020', 'value': '801,00', 'statusCode': 'OK'},
                ]
            }
        }

        df = json_to_dataframe(json_data)

        assert df.index[0] == pd.Timestamp('2020-01-01')
        assert df.index[1] == pd.Timestamp('2020-01-02')
        assert df.index[2] == pd.Timestamp('2020-01-03')

    def test_verbose_mode(self, capsys):
        """Verifica que verbose mode imprima información."""
        json_data = {
            'Series': {
                'Obs': [
                    {'indexDateString': '01-01-2020', 'value': '800,50', 'statusCode': 'OK'},
                ]
            }
        }

        json_to_dataframe(json_data, verbose=True)
        captured = capsys.readouterr()

        assert "DataFrame creado" in captured.out


class TestSaveToCsv:
    """Tests para la función save_to_csv."""

    def test_save_to_custom_path(self, sample_bcch_data):
        """Verifica guardado en ruta personalizada."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_output.csv')

            saved_path = save_to_csv(sample_bcch_data, output_path)

            assert saved_path == output_path
            assert os.path.exists(output_path)

    def test_csv_content_integrity(self, sample_bcch_data):
        """Verifica integridad del contenido guardado."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_output.csv')

            save_to_csv(sample_bcch_data, output_path)
            loaded_df = pd.read_csv(output_path, index_col=0, parse_dates=True)

            # Verificar dimensiones
            assert loaded_df.shape == sample_bcch_data.shape
            # Verificar columnas
            assert list(loaded_df.columns) == list(sample_bcch_data.columns)
            # Verificar valores (permitiendo diferencias menores)
            assert loaded_df['Valor'].equals(sample_bcch_data['Valor']) or \
                   (loaded_df['Valor'] - sample_bcch_data['Valor']).abs().max() < 0.01

    def test_creates_directory_if_not_exists(self):
        """Verifica que cree directorios si no existen."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'new_folder', 'test.csv')

            df = pd.DataFrame({'A': [1, 2, 3]})
            save_to_csv(df, output_path)

            assert os.path.exists(output_path)

    def test_verbose_mode(self, sample_bcch_data, capsys):
        """Verifica que verbose mode imprima ruta."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.csv')

            save_to_csv(sample_bcch_data, output_path, verbose=True)
            captured = capsys.readouterr()

            assert "Archivo guardado en" in captured.out
            assert output_path in captured.out


class TestProcessAndSave:
    """Tests para el pipeline completo process_and_save."""

    def test_full_pipeline(self):
        """Verifica el pipeline completo de procesamiento."""
        json_data = {
            'Series': {
                'Obs': [
                    {'indexDateString': '01-01-2020', 'value': '800,50', 'statusCode': 'OK'},
                    {'indexDateString': '02-01-2020', 'value': '801,25', 'statusCode': 'OK'},
                ]
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.csv')

            df = process_and_save(json_data, output_path, verbose=False)

            assert df is not None
            assert len(df) == 2
            assert os.path.exists(output_path)

    def test_pipeline_with_empty_data(self):
        """Verifica manejo de datos vacíos en el pipeline."""
        json_data = {'Series': {}}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test.csv')

            df = process_and_save(json_data, output_path)

            assert df is None
            # No debe crear archivo si no hay datos
            assert not os.path.exists(output_path)
