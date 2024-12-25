import logging
from typing import Dict
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import numpy as np

from src.preprocessing import DataProcessor
from src.model import ModelSelector
from src.visualization import ModelVisualizer
from src.utils import load_config

logger = logging.getLogger(__name__)

def train_model():
    """Train models for each region."""
    config = load_config()
    preprocessor = DataProcessor(config)
    visualizer = ModelVisualizer(config)
    trained_models = {}
    metrics_by_region = {}
    
    # Load data
    data = pd.read_csv(config['data']['input_path'])
    
    # Preprocess data
    data_by_region = preprocessor.prepare_data(data)
    
    for region, region_data in data_by_region.items():
        try:
            # Unpack the tuple values
            features, target = region_data
            
            # Convert datetime to numeric (timestamp)
            if 'Week Ending Date' in features.columns:
                features['Week Ending Date'] = pd.to_datetime(features['Week Ending Date']).astype(np.int64) // 10**9
            
            # Convert categorical columns to numeric
            categorical_columns = features.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                features[col] = pd.Categorical(features[col]).codes
            
            # Split into train/test sets
            train_features, test_features, train_target, test_target = train_test_split(
                features, target, test_size=0.2, random_state=42
            )
            
            # Get feature names
            feature_names = features.columns.tolist()
            
            # Train model selector
            model_selector = ModelSelector(config)
            result = model_selector.select_best_model(
                train_features,
                train_target,
                test_features,
                test_target,
                feature_names
            )
            
            # Store the trained model and metrics
            trained_models[region] = result
            metrics_by_region[region] = result['metrics']
            
            # Create actual vs predicted visualization
            y_pred = result['model'].predict(test_features)
            visualizer.plot_actual_vs_predicted(
                region,
                test_target,
                y_pred,
                test_features.index,
                test_features['Week Ending Date'] if 'Week Ending Date' in test_features.columns else None
            )
            
        except Exception as e:
            logger.error(f"Error training models for region {region}: {str(e)}")
            continue
    
    if not trained_models:
        raise ValueError("No models were successfully trained for any region")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame.from_dict(metrics_by_region, orient='index')
    metrics_df.index.name = 'region'
    metrics_path = Path(config['output']['model_comparison_path'])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path)
    logger.info(f"Model metrics saved to {metrics_path}")
    
    # Create performance summary visualization
    visualizer.create_performance_summary(metrics_by_region)
    
    # Save trained models
    save_path = Path(config['output']['model_path'])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained_models, save_path)
    logger.info(f"Models saved to {save_path}")
    
    return trained_models

if __name__ == "__main__":
    train_model() 