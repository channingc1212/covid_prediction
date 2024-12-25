from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from typing import Dict, Tuple, List, Any
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import logging

logger = logging.getLogger(__name__)

class BaseModel:
    def __init__(self, config: dict):
        self.config = config
        
    def train(self, train_data: pd.DataFrame, train_target: pd.Series) -> None:
        raise NotImplementedError
        
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
        
    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics with robust MAPE calculation."""
        # Calculate MAE and RMSE as before
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Calculate MAPE with handling for edge cases
        def calculate_mape(y_true, y_pred):
            """
            Calculate MAPE with handling for edge cases:
            - Exclude cases where actual value is 0
            - Cap individual percentage errors at 100%
            - Return mean of valid percentage errors
            """
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            # Mask for non-zero actual values
            non_zero_mask = np.abs(y_true) > 1e-10
            
            if not any(non_zero_mask):
                logger.warning("All actual values are zero or very close to zero. MAPE calculation may be unreliable.")
                return np.nan
                
            # Calculate percentage errors for non-zero actual values
            percentage_errors = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]) * 100
            
            # Cap individual errors at 100%
            percentage_errors = np.minimum(percentage_errors, 100)
            
            return np.mean(percentage_errors)
        
        mape = calculate_mape(y_true, y_pred)
        
        # Log warning if MAPE is high
        if mape > 50:  # You can adjust this threshold
            logger.warning(f"High MAPE value ({mape:.2f}%) detected. This might indicate prediction issues.")
            
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }

class ProphetModel(BaseModel):
    def train(self, train_data: pd.DataFrame, train_target: pd.Series) -> None:
        self.model = Prophet(**self.config['model']['prophet_params'])
        df = pd.DataFrame({'ds': train_data.index, 'y': train_target})
        self.model.fit(df)
        
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        future = pd.DataFrame({'ds': features.index})
        forecast = self.model.predict(future)
        return forecast['yhat'].values

class RandomForestModel(BaseModel):
    def train(self, train_data: pd.DataFrame, train_target: pd.Series) -> None:
        self.model = RandomForestRegressor(**self.config['model']['random_forest_params'])
        self.model.fit(train_data, train_target)
        
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.model.predict(features)

class XGBoostModel(BaseModel):
    def train(self, train_data: pd.DataFrame, train_target: pd.Series) -> None:
        self.model = xgb.XGBRegressor(**self.config['model']['xgboost_params'])
        self.model.fit(train_data, train_target)
        
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return self.model.predict(features)

class ModelSelector:
    def __init__(self, config):
        self.config = config
        
    def _create_model(self, model_name):
        """Create a model instance based on the model name."""
        if model_name == 'random_forest':
            return RandomForestRegressor(
                n_estimators=self.config['model']['random_forest_params']['n_estimators'],
                max_depth=self.config['model']['random_forest_params']['max_depth'],
                random_state=42
            )
        elif model_name == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=self.config['model']['xgboost_params']['n_estimators'],
                max_depth=self.config['model']['xgboost_params']['max_depth'],
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_name}")

    def select_best_model(self, train_features, train_target, test_features, test_target, feature_names):
        """Train and evaluate multiple models, select the best one based on MAPE."""
        best_metrics = None
        best_model = None
        best_model_name = None
        
        # Validate input data
        if not self._validate_data(train_features, train_target, test_features, test_target):
            raise ValueError("Invalid input data detected")
        
        for model_name in self.config['model']['models_to_try']:
            try:
                logger.info(f"Training {model_name}...")
                model = self._create_model(model_name)
                
                # Additional check for data validity
                if train_features.isna().any().any() or train_target.isna().any():
                    logger.error(f"NaN values found in training data for {model_name}")
                    continue
                    
                if test_features.isna().any().any() or test_target.isna().any():
                    logger.error(f"NaN values found in test data for {model_name}")
                    continue
                
                # Train the model
                model.fit(train_features, train_target)
                
                # Make predictions
                predictions = model.predict(test_features)
                
                # Validate predictions
                if not self._validate_predictions(predictions):
                    logger.error(f"Invalid predictions from {model_name}")
                    continue
                
                # Calculate metrics
                try:
                    metrics = {
                        'mape': self._calculate_mape(test_target, predictions),
                        'mae': mean_absolute_error(test_target, predictions),
                        'rmse': np.sqrt(mean_squared_error(test_target, predictions))
                    }
                    
                    logger.info(f"{model_name} metrics: {metrics}")
                    
                    # Update best model if this one is better (using MAPE as criterion)
                    if best_metrics is None or metrics['mape'] < best_metrics['mape']:
                        best_metrics = metrics
                        best_model = model
                        best_model_name = model_name
                        
                except Exception as e:
                    logger.error(f"Error calculating metrics for {model_name}: {str(e)}")
                    continue
                    
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        if best_model is None:
            raise ValueError("No models were successfully trained")
            
        logger.info(f"Selected best model: {best_model_name}")
        return {
            'model': best_model,
            'name': best_model_name,
            'metrics': best_metrics
        }
        
    def _validate_data(self, train_features, train_target, test_features, test_target):
        """Validate input data for model training"""
        try:
            # Check for NaN or infinite values
            if (train_features.isna().any().any() or 
                train_target.isna().any() or 
                test_features.isna().any().any() or 
                test_target.isna().any()):
                logger.error("NaN values found in input data")
                return False
                
            if (np.isinf(train_features.values).any() or 
                np.isinf(train_target.values).any() or 
                np.isinf(test_features.values).any() or 
                np.isinf(test_target.values).any()):
                logger.error("Infinite values found in input data")
                return False
                
            # Check for reasonable data sizes
            if len(train_features) < 10 or len(test_features) < 5:
                logger.error("Insufficient data points for training/testing")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            return False
            
    def _validate_predictions(self, predictions):
        """Validate model predictions"""
        try:
            if np.isnan(predictions).any():
                logger.error("NaN values in predictions")
                return False
                
            if np.isinf(predictions).any():
                logger.error("Infinite values in predictions")
                return False
                
            if len(predictions) == 0:
                logger.error("Empty predictions")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating predictions: {str(e)}")
            return False
            
    def _calculate_mape(self, y_true, y_pred):
        """Calculate MAPE with proper handling of edge cases"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        # Remove pairs where true value is zero or close to zero
        non_zero_mask = np.abs(y_true) > 1e-10
        if not any(non_zero_mask):
            logger.warning("All actual values are zero or very close to zero")
            return np.nan
            
        # Calculate percentage errors
        percentage_errors = np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]) * 100
        
        # Cap extreme errors at 100%
        percentage_errors = np.minimum(percentage_errors, 100)
        
        return np.mean(percentage_errors) 