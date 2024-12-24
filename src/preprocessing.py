import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.feature_columns = []

    def prepare_data(self, data: pd.DataFrame) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """Prepare data for modeling including preprocessing and feature engineering"""
        try:
            data = data.copy()
            
            # Convert date column to datetime
            date_col = self.config['data']['date_column']
            target_col = self.config['data']['target_column']
            group_col = self.config['data']['group_by_column']
            
            data[date_col] = pd.to_datetime(data[date_col])
            
            # Group by geographic aggregation
            grouped_data = {}
            for group_name, group_df in data.groupby(group_col):
                features = self._create_features(group_df)
                target = group_df[target_col]
                
                # Add diagnostic information
                logger.info(f"Group {group_name}:")
                logger.info(f"  - Total rows: {len(group_df)}")
                logger.info(f"  - Missing values in target: {target.isna().sum()}")
                logger.info(f"  - Missing values in features: {features.isna().sum().sum()}")
                
                grouped_data[group_name] = (features, target)
                
            logger.info(f"Prepared data for {len(grouped_data)} groups")
            return grouped_data
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features for modeling"""
        df = df.copy()
        
        # Create time features
        df = self._create_time_features(df)
        
        # Create lag features
        df = self._create_lag_features(df)
        
        # Create rolling features
        df = self._create_rolling_features(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        return df

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        date_col = self.config['data']['date_column']
        
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['quarter'] = df[date_col].dt.quarter
        df['is_weekend'] = df[date_col].dt.dayofweek.isin([5, 6]).astype(int)
        
        self.feature_columns.extend(['year', 'month', 'day_of_week', 'quarter', 'is_weekend'])
        return df

    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features"""
        df = df.copy()
        target_col = self.config['data']['target_column']
        
        for lag in self.config['data']['feature_engineering']['lag_features']:
            col_name = f'lag_{lag}'
            df[col_name] = df[target_col].shift(lag)
            self.feature_columns.append(col_name)
        return df

    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features"""
        df = df.copy()
        target_col = self.config['data']['target_column']
        
        for window in self.config['data']['feature_engineering']['rolling_features']['windows']:
            for operation in self.config['data']['feature_engineering']['rolling_features']['operations']:
                col_name = f'rolling_{window}_{operation}'
                if operation == 'mean':
                    df[col_name] = df[target_col].rolling(window=window).mean()
                elif operation == 'std':
                    df[col_name] = df[target_col].rolling(window=window).std()
                elif operation == 'min':
                    df[col_name] = df[target_col].rolling(window=window).min()
                elif operation == 'max':
                    df[col_name] = df[target_col].rolling(window=window).max()
                self.feature_columns.append(col_name)
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        df = df.copy()
        
        for column in df.columns:
            if df[column].isna().any():
                method = self.config['data']['feature_engineering']['missing_value_handling']['method']
                max_gap = self.config['data']['feature_engineering']['missing_value_handling']['max_gap_to_fill']
                
                if method == "interpolate":
                    interp_method = self.config['data']['feature_engineering']['missing_value_handling']['interpolation_method']
                    df[column] = df[column].interpolate(method=interp_method, limit=max_gap)
                elif method == "forward_fill":
                    df[column] = df[column].fillna(method='ffill', limit=max_gap)
                elif method == "backward_fill":
                    df[column] = df[column].fillna(method='bfill', limit=max_gap)
                
        return df

    # ... [rest of the feature engineering methods] 