import pandas as pd
import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        
    def load_data(self) -> pd.DataFrame:
        """Load and validate the input data."""
        try:
            df = pd.read_csv(self.config['data']['input_path'])
            
            # Ensure required columns exist
            required_cols = [self.config['data']['date_column'], 
                           self.config['data']['target_column']]
            assert all(col in df.columns for col in required_cols)
            
            # Convert date column to datetime
            df[self.config['data']['date_column']] = pd.to_datetime(
                df[self.config['data']['date_column']]
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in time series data."""
        df = df.copy()
        
        # Sort by date
        df = df.sort_values('ds')
        
        # Check for gaps in time series
        time_diff = df['ds'].diff()
        if time_diff.nunique() > 1:
            logger.warning(f"Irregular time intervals detected. Time differences: {time_diff.unique()}")
            
        # Get missing value handling parameters
        method = self.config['data']['missing_value_handling']['method']
        max_gap = self.config['data']['missing_value_handling']['max_gap_to_fill']
        
        # Identify gaps longer than max_gap
        long_gaps = df['ds'].diff() > pd.Timedelta(days=max_gap)
        if long_gaps.any():
            gap_locations = df[long_gaps].index
            logger.warning(f"Large gaps found at: {df.loc[gap_locations, 'ds'].tolist()}")
        
        # Handle missing values based on configuration
        if method == "interpolate":
            interp_method = self.config['data']['missing_value_handling']['interpolation_method']
            df['y'] = df['y'].interpolate(
                method=interp_method,
                limit=max_gap,
                limit_direction='both'
            )
        elif method == "forward_fill":
            df['y'] = df['y'].fillna(method='ffill', limit=max_gap)
        elif method == "backward_fill":
            df['y'] = df['y'].fillna(method='bfill', limit=max_gap)
        elif method == "mean":
            df['y'] = df['y'].fillna(df['y'].mean())
            
        # Log remaining missing values
        remaining_missing = df['y'].isna().sum()
        if remaining_missing > 0:
            logger.warning(f"Remaining missing values after handling: {remaining_missing}")
            
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare data for Prophet model."""
        try:
            # Prophet requires columns named 'ds' and 'y'
            prophet_df = df.rename(columns={
                self.config['data']['date_column']: 'ds',
                self.config['data']['target_column']: 'y'
            })
            
            # Handle missing values
            prophet_df = self.handle_missing_values(prophet_df)
            
            # Sort by date
            prophet_df = prophet_df.sort_values('ds')
            
            # Split into train and test
            train_size = int(len(prophet_df) * (1 - self.config['data']['train_test_split']))
            train_df = prophet_df.iloc[:train_size]
            test_df = prophet_df.iloc[train_size:]
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise 