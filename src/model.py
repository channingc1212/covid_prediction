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
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
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
        """Train and evaluate multiple models, select the best one."""
        best_metrics = None
        best_model = None
        best_model_name = None
        
        for model_name in self.config['model']['models_to_try']:
            try:
                model = self._create_model(model_name)
                model.fit(train_features, train_target)
                
                # Make predictions
                predictions = model.predict(test_features)
                
                # Calculate metrics
                metrics = {
                    'mae': mean_absolute_error(test_target, predictions),
                    'rmse': np.sqrt(mean_squared_error(test_target, predictions)),
                    'mape': mean_absolute_percentage_error(test_target, predictions) * 100
                }
                
                logger.info(f"{model_name} metrics: {metrics}")
                
                # Update best model if this one is better (using RMSE as criterion)
                if best_metrics is None or metrics['rmse'] < best_metrics['rmse']:
                    best_metrics = metrics
                    best_model = model
                    best_model_name = model_name
                    
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        if best_model is None:
            raise ValueError("No models were successfully trained")
            
        return {
            'model': best_model,
            'name': best_model_name,
            'metrics': best_metrics
        } 