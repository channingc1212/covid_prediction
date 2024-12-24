import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.preprocessing import DataProcessor
from src.model import ModelSelector
from src.utils import load_config
import yaml
import os

@pytest.fixture
def sample_config():
    return {
        'data': {
            'input_path': 'data/raw_weekly_hospital_respiratory_data_2020_2024.csv',
            'date_column': 'date',
            'target_column': 'value',
            'group_by_column': 'region',
            'train_test_split': 0.2,
            'feature_engineering': {
                'enabled': True,
                'time_features': {
                    'enabled': True,
                    'features': ['year', 'month', 'day_of_week', 'quarter', 'is_weekend']
                },
                'lag_features': [1, 7],
                'rolling_features': {
                    'windows': [7],
                    'operations': ['mean', 'std']
                },
                'missing_value_handling': {
                    'method': 'interpolate',
                    'interpolation_method': 'linear',
                    'limit': 7,
                    'max_gap_to_fill': 7
                }
            }
        },
        'model': {
            'models_to_try': ['random_forest', 'xgboost'],
            'random_forest_params': {
                'n_estimators': 100,
                'max_depth': 10
            },
            'xgboost_params': {
                'n_estimators': 100,
                'max_depth': 7
            }
        },
        'output': {
            'model_path': 'models/trained_models.joblib',
            'visualizations_dir': 'visualizations'
        }
    }

@pytest.fixture
def sample_data():
    # Create sample time series data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    regions = ['MA', 'NY']
    
    data = []
    for region in regions:
        for date in dates:
            value = np.random.normal(100, 10)
            data.append({
                'date': date,
                'region': region,
                'value': value
            })
    
    return pd.DataFrame(data)

@pytest.fixture(autouse=True)
def setup_and_teardown(sample_config):
    """Create temporary config file before tests and clean up after"""
    # Create config directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # Write temporary config file
    with open('config.yaml', 'w') as f:
        yaml.dump(sample_config, f)
    
    yield
    
    # Clean up
    if os.path.exists('config.yaml'):
        os.remove('config.yaml')

def test_data_processor(sample_config, sample_data):
    processor = DataProcessor(sample_config)
    grouped_data = processor.prepare_data(sample_data)
    
    # Check if data is properly grouped
    assert len(grouped_data) == 2
    assert 'MA' in grouped_data
    assert 'NY' in grouped_data
    
    # Check features
    features, target = grouped_data['MA']
    assert all(col in features.columns for col in processor.feature_columns)
    assert len(target) == len(features)

def test_model_training(sample_config, sample_data):
    processor = DataProcessor(sample_config)
    grouped_data = processor.prepare_data(sample_data)
    
    features, target = grouped_data['MA']
    
    # Only select numeric columns for training
    numeric_columns = features.select_dtypes(include=['int64', 'float64']).columns
    features = features[numeric_columns]
    
    train_size = int(len(features) * 0.8)
    
    train_features = features[:train_size]
    test_features = features[train_size:]
    train_target = target[:train_size]
    test_target = target[train_size:]
    
    model_selector = ModelSelector(sample_config)
    best_model, metrics = model_selector.select_best_model(
        train_features, train_target,
        test_features, test_target,
        numeric_columns.tolist()
    )
    
    assert best_model is not None
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert 'mape' in metrics

def test_end_to_end_pipeline(sample_config, sample_data):
    # Save sample data
    os.makedirs('data', exist_ok=True)
    sample_data.to_csv('data/raw_weekly_hospital_respiratory_data_2020_2024.csv', index=False)
    
    # Train models
    from src.train import train_model
    models = train_model()
    
    # Verify models were trained
    assert len(models) > 0
    assert all(key in ['MA', 'NY'] for key in models.keys())
    
    # Clean up
    if os.path.exists('data/raw_weekly_hospital_respiratory_data_2020_2024.csv'):
        os.remove('data/raw_weekly_hospital_respiratory_data_2020_2024.csv')
    if os.path.exists('models/trained_models.joblib'):
        os.remove('models/trained_models.joblib') 