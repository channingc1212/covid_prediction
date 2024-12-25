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
            
            # Handle extreme values in target column
            data = self._handle_extreme_values(data, target_col)
            
            # Group by geographic aggregation
            grouped_data = {}
            for group_name, group_df in data.groupby(group_col):
                try:
                    # Create features
                    features = self._create_features(group_df)
                    target = group_df[target_col]
                    
                    # Handle missing and infinite values
                    features, target = self._clean_data(features, target)
                    
                    if len(features) < 10:  # Minimum required data points
                        logger.warning(f"Insufficient data points for {group_name} after cleaning. Skipping.")
                        continue
                        
                    # Add diagnostic information
                    logger.info(f"Group {group_name}:")
                    logger.info(f"  - Total rows: {len(group_df)}")
                    logger.info(f"  - Missing values in target: {target.isna().sum()}")
                    logger.info(f"  - Missing values in features: {features.isna().sum().sum()}")
                    logger.info(f"  - Final shape after cleaning: {features.shape}")
                    
                    grouped_data[group_name] = (features, target)
                    
                except Exception as e:
                    logger.error(f"Error processing group {group_name}: {str(e)}")
                    continue
                
            if not grouped_data:
                raise ValueError("No valid groups after data cleaning")
                
            logger.info(f"Prepared data for {len(grouped_data)} groups")
            return grouped_data
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise

    def _handle_extreme_values(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Handle extreme values using statistical methods"""
        df = df.copy()
        
        # Calculate statistics for the column
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # Log extreme values
        extreme_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        if extreme_mask.any():
            logger.warning(f"Found {extreme_mask.sum()} extreme values in {column}")
            logger.info(f"Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        # Cap extreme values
        df.loc[df[column] < lower_bound, column] = lower_bound
        df.loc[df[column] > upper_bound, column] = upper_bound
        
        return df

    def _clean_data(self, features: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Clean features and target data"""
        # Get common valid indices
        valid_features = ~(features.isna().any(axis=1) | features.isin([np.inf, -np.inf]).any(axis=1))
        valid_target = ~(target.isna() | target.isin([np.inf, -np.inf]))
        valid_indices = valid_features & valid_target
        
        if not valid_indices.any():
            raise ValueError("No valid data points after cleaning")
        
        # Filter both features and target
        features = features[valid_indices].copy()
        target = target[valid_indices].copy()
        
        return features, target

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
        """Handle missing values in features and target"""
        df = df.copy()
        
        # First forward fill
        df = df.fillna(method='ffill')
        # Then backward fill any remaining NaNs
        df = df.fillna(method='bfill')
        
        # If any NaNs remain, fill with column mean
        if df.isna().any().any():
            df = df.fillna(df.mean())
        
        return df

    # ... [rest of the feature engineering methods] 