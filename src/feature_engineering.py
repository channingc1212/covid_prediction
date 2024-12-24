import pandas as pd
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, config: dict):
        self.config = config
        self.feature_columns = []  # Track created feature columns
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from datetime column."""
        if not self.config['data']['feature_engineering']['time_features']['enabled']:
            return df
            
        df = df.copy()
        time_features = self.config['data']['feature_engineering']['time_features']['features']
        
        feature_mapping = {
            'year': lambda x: x.dt.year,
            'month': lambda x: x.dt.month,
            'day_of_week': lambda x: x.dt.dayofweek,
            'day_of_year': lambda x: x.dt.dayofyear,
            'quarter': lambda x: x.dt.quarter,
            'is_weekend': lambda x: x.dt.dayofweek.isin([5, 6]).astype(int)
        }
        
        for feature in time_features:
            if feature in feature_mapping:
                df[feature] = feature_mapping[feature](df['ds'])
                self.feature_columns.append(feature)
                
        return df
        
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features."""
        df = df.copy()
        for lag in self.config['data']['feature_engineering']['lag_features']:
            col_name = f'lag_{lag}'
            df[col_name] = df['y'].shift(lag)
            self.feature_columns.append(col_name)
        return df
        
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features."""
        df = df.copy()
        windows = self.config['data']['feature_engineering']['rolling_features']['windows']
        operations = self.config['data']['feature_engineering']['rolling_features']['operations']
        
        operation_mapping = {
            'mean': lambda x: x.mean(),
            'std': lambda x: x.std(),
            'min': lambda x: x.min(),
            'max': lambda x: x.max()
        }
        
        for window in windows:
            for op in operations:
                if op in operation_mapping:
                    col_name = f'rolling_{op}_{window}'
                    df[col_name] = df['y'].rolling(window=window).agg(operation_mapping[op])
                    self.feature_columns.append(col_name)
                    
        return df
        
    def add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add any custom features specified in config."""
        custom_features = self.config['data']['feature_engineering']['custom_features']
        if custom_features:
            self.feature_columns.extend(custom_features)
        return df
        
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        if not self.config['data']['feature_engineering']['enabled']:
            logger.info("Feature engineering disabled, using raw features only")
            return df
            
        df = self.create_time_features(df)
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.add_custom_features(df)
        
        # Handle missing values in created features
        for column in self.feature_columns:
            if df[column].isna().any():
                method = self.config['data']['missing_value_handling']['method']
                max_gap = self.config['data']['missing_value_handling']['max_gap_to_fill']
                
                if method == "interpolate":
                    interp_method = self.config['data']['missing_value_handling']['interpolation_method']
                    df[column] = df[column].interpolate(
                        method=interp_method,
                        limit=max_gap,
                        limit_direction='both'
                    )
                elif method == "forward_fill":
                    df[column] = df[column].fillna(method='ffill', limit=max_gap)
                elif method == "backward_fill":
                    df[column] = df[column].fillna(method='bfill', limit=max_gap)
                elif method == "mean":
                    df[column] = df[column].fillna(df[column].mean())
        
        # Log created features and any remaining missing values
        logger.info(f"Created features: {self.feature_columns}")
        missing_counts = df[self.feature_columns].isna().sum()
        if missing_counts.any():
            logger.warning(f"Remaining missing values in features: {missing_counts[missing_counts > 0]}")
        
        return df
        
    def get_feature_columns(self) -> List[str]:
        """Return list of feature columns to use for modeling."""
        return self.feature_columns 