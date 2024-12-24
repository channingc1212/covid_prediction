from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from typing import Dict, Union, List
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import joblib
from itertools import product

logger = logging.getLogger(__name__)

class ModelFactory:
    @staticmethod
    def create_model(model_type: str, config: dict) -> 'BaseTimeSeriesModel':
        if model_type == "prophet":
            return ProphetModel(config)
        elif model_type == "arima":
            return ARIMAModel(config)
        elif model_type == "random_forest":
            return RandomForestModel(config)
        elif model_type == "xgboost":
            return XGBoostModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

class BaseTimeSeriesModel:
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        
    def train(self, train_data: pd.DataFrame) -> None:
        raise NotImplementedError
        
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
        
    def evaluate(self, test_data: pd.DataFrame, predictions: np.ndarray) -> Dict[str, float]:
        metrics = {
            'mae': mean_absolute_error(test_data['y'], predictions),
            'rmse': np.sqrt(mean_squared_error(test_data['y'], predictions)),
            'mape': np.mean(np.abs((test_data['y'] - predictions) / test_data['y'])) * 100
        }
        return metrics

class ProphetModel(BaseTimeSeriesModel):
    def train(self, train_data: pd.DataFrame) -> None:
        self.model = Prophet(**self.config['model']['prophet_params'])
        self.model.fit(train_data[['ds', 'y']])
        
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        forecast = self.model.predict(data[['ds']])
        return forecast['yhat'].values

class ARIMAModel(BaseTimeSeriesModel):
    def train(self, train_data: pd.DataFrame) -> None:
        best_aic = float('inf')
        best_params = None
        
        # Grid search over ARIMA parameters
        p = self.config['model']['arima_params']['p']
        d = self.config['model']['arima_params']['d']
        q = self.config['model']['arima_params']['q']
        
        for order in product(p, d, q):
            try:
                model = SARIMAX(train_data['y'], order=order)
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_params = order
                    self.model = results
            except:
                continue
                
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return self.model.forecast(len(data))

class RandomForestModel(BaseTimeSeriesModel):
    def train(self, train_data: pd.DataFrame) -> None:
        self.model = RandomForestRegressor(**self.config['model']['random_forest_params'])
        # Use feature columns from feature engineering
        self.feature_columns = [col for col in train_data.columns 
                              if col not in ['ds', 'y'] and not pd.isna(train_data[col]).any()]
        logger.info(f"Using features: {self.feature_columns}")
        self.model.fit(train_data[self.feature_columns], train_data['y'])
        
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return self.model.predict(data[self.feature_columns])

class XGBoostModel(BaseTimeSeriesModel):
    def train(self, train_data: pd.DataFrame) -> None:
        self.model = xgb.XGBRegressor(**self.config['model']['xgboost_params'])
        features = [col for col in train_data.columns if col not in ['ds', 'y']]
        self.model.fit(train_data[features], train_data['y'])
        self.feature_columns = features
        
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return self.model.predict(data[self.feature_columns])

class ModelSelector:
    def __init__(self, config: dict):
        self.config = config
        
    def select_best_model(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple:
        results = []
        
        for model_type in self.config['model']['models_to_try']:
            logger.info(f"Training {model_type} model...")
            model = ModelFactory.create_model(model_type, self.config)
            
            try:
                model.train(train_data)
                predictions = model.predict(test_data)
                metrics = model.evaluate(test_data, predictions)
                
                results.append({
                    'model_type': model_type,
                    'model': model,
                    'metrics': metrics
                })
                
                logger.info(f"{model_type} metrics: {metrics}")
                
            except Exception as e:
                logger.error(f"Error training {model_type}: {str(e)}")
                continue
        
        # Select best model based on RMSE
        best_result = min(results, key=lambda x: x['metrics']['rmse'])
        
        # Save comparison results
        comparison_df = pd.DataFrame([
            {**{'model': r['model_type']}, **r['metrics']}
            for r in results
        ])
        comparison_df.to_csv(self.config['output']['model_comparison_path'], index=False)
        
        return best_result['model'], best_result['metrics'] 