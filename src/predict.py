import joblib
import logging
import pandas as pd
from src.preprocessing import DataProcessor
from src.utils import load_config

logger = logging.getLogger(__name__)

def predict(input_data: pd.DataFrame, group: str):
    """Make predictions for a specific group"""
    try:
        # Load config and models
        config = load_config()
        models = joblib.load(config['output']['model_path'])
        
        if group not in models:
            raise ValueError(f"No model found for group: {group}")
            
        model_info = models[group]
        
        # Process input data
        processor = DataProcessor(config)
        features = processor._create_features(input_data)
        
        # Make prediction
        prediction = model_info['model'].predict(features)
        logger.info(f"Prediction made successfully for group {group}")
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise