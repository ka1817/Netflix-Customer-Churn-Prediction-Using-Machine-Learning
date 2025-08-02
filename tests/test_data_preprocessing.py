import pytest
import pandas as pd
from src.data_preprocessing import DataPreprocessor

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'watch_hours': [10, 20],
        'last_login_days': [1, 5],
        'number_of_profiles': [2, 3],
        'avg_watch_time_per_day': [2.5, 3.0],
        'subscription_type': ['basic', 'premium'],
        'payment_method': ['credit', 'paypal'],
        'churned': [0, 1]
    })

def test_preprocessing_pipeline_builds(sample_df):
    num_features = ['watch_hours', 'last_login_days', 'number_of_profiles', 'avg_watch_time_per_day']
    cat_features = ['subscription_type', 'payment_method']
    preprocessor = DataPreprocessor(num_features, cat_features)
    preprocessor.build_pipeline()
    assert preprocessor.pipeline is not None, "Pipeline should be built"

def test_transform_output_shape(sample_df):
    num_features = ['watch_hours', 'last_login_days', 'number_of_profiles', 'avg_watch_time_per_day']
    cat_features = ['subscription_type', 'payment_method']
    preprocessor = DataPreprocessor(num_features, cat_features)
    preprocessor.build_pipeline()
    X = sample_df.drop(columns=['churned'])
    transformed = preprocessor.pipeline.fit_transform(X)
    assert transformed.shape[0] == sample_df.shape[0], "Rows should remain same after transform"
