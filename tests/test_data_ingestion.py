import pytest
from src.data_ingestion import DataIngestion
import pandas as pd

def test_data_ingestion_loads_csv():
    ingestion = DataIngestion()
    df = ingestion.load_data()
    assert isinstance(df, pd.DataFrame), "Loaded data should be a pandas DataFrame"
    assert df.shape[0] > 0, "Dataframe should not be empty"

def test_data_ingestion_columns():
    ingestion = DataIngestion()
    df = ingestion.load_data()
    expected_columns = ['age', 'watch_hours', 'last_login_days', 'churned']
    for col in expected_columns:
        assert col in df.columns, f"{col} should exist in the dataframe"
