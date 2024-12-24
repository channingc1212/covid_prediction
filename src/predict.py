import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_model(model_path="models/model.joblib"):
    """Load the trained model from disk"""
    try:
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def predict(input_data):
    """
    Make predictions using the trained model
    
    Args:
        input_data (dict): Dictionary containing feature values
        
    Returns:
        Prediction result
    """
    try:
        # Load the model
        model = load_model()
        
        # Convert input dictionary to format expected by model
        # Modify this part according to your model's requirements
        features = [
            input_data["feature1"],
            input_data["feature2"],
            # Add other features in the order expected by your model
        ]
        
        # Make prediction
        prediction = model.predict([features])[0]
        logger.info("Prediction made successfully")
        return prediction
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise 