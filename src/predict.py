import joblib
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from src.preprocessing import DataProcessor
from src.utils import load_config

logger = logging.getLogger(__name__)

def generate_future_dates(last_date: pd.Timestamp, periods: int = 52) -> pd.DataFrame:
    """Generate future dates for prediction"""
    future_dates = pd.date_range(
        start=last_date + timedelta(weeks=1),
        periods=periods,
        freq='W-SAT'  # Weekly frequency ending on Saturday
    )
    return pd.DataFrame({
        'Week Ending Date': future_dates
    })

def prepare_future_features(df: pd.DataFrame, processor: DataProcessor) -> pd.DataFrame:
    """Prepare features for future prediction"""
    try:
        logger.info(f"Preparing features for DataFrame with shape: {df.shape}")
        
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Ensure date column is in datetime format
        date_col = processor.config['data']['date_column']
        target_col = processor.config['data']['target_column']
        group_col = processor.config['data']['group_by_column']
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Log the date range
        logger.info(f"Date range in input: {df[date_col].min()} to {df[date_col].max()}")
        
        # Convert all numeric columns to float32 first
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_columns:
            if col != date_col:
                df[col] = df[col].astype('float32')
        
        # Create features DataFrame
        features = pd.DataFrame(index=df.index)
        
        # Create time features directly as numeric
        features['year'] = df[date_col].dt.year.astype('float32')
        features['month'] = df[date_col].dt.month.astype('float32')
        features['day_of_week'] = df[date_col].dt.dayofweek.astype('float32')
        features['quarter'] = df[date_col].dt.quarter.astype('float32')
        features['is_weekend'] = df[date_col].dt.dayofweek.isin([5, 6]).astype('float32')
        
        # Copy and convert other columns
        for col in df.columns:
            if col not in [date_col] and col not in features.columns:
                if df[col].dtype == 'object' or df[col].dtype == 'category':
                    # Convert categorical to numeric
                    features[col] = pd.Categorical(df[col]).codes.astype('float32')
                else:
                    # Convert numeric
                    features[col] = df[col].astype('float32')
        
        logger.info(f"Created initial features. Shape: {features.shape}")
        
        # For lag and rolling features, we need the target column
        if target_col not in features.columns:
            # Get the last known value from the historical data
            historical_data = df[df[target_col].notna()]
            if not historical_data.empty:
                last_known_value = historical_data[target_col].iloc[-1]
                logger.info(f"Using last known value for target: {last_known_value}")
                features[target_col] = float(last_known_value)
            else:
                logger.error("No historical data found for target column")
                raise ValueError("Cannot generate features without historical data")
        
        # Create lag features
        features = processor._create_lag_features(features)
        logger.info(f"Created lag features. Shape: {features.shape}")
        
        # Create rolling features
        features = processor._create_rolling_features(features)
        logger.info(f"Created rolling features. Shape: {features.shape}")
        
        # Handle missing values
        features = processor._handle_missing_values(features)
        logger.info(f"Handled missing values. Shape: {features.shape}")
        
        # Drop unnecessary columns
        columns_to_drop = [date_col, target_col, group_col]
        features = features.drop(columns=columns_to_drop, errors='ignore')
        
        # Final conversion to ensure all columns are float32
        for col in features.columns:
            if not pd.api.types.is_numeric_dtype(features[col]):
                logger.warning(f"Converting non-numeric column {col} to float32")
                features[col] = pd.to_numeric(features[col], errors='coerce').astype('float32')
            else:
                features[col] = features[col].astype('float32')
        
        logger.info(f"Final features shape: {features.shape}")
        logger.info(f"Feature columns and types:")
        for col in features.columns:
            logger.info(f"  - {col}: {features[col].dtype}")
        
        # Verify no non-numeric columns remain
        non_numeric = features.select_dtypes(exclude=['float32', 'float64']).columns
        if len(non_numeric) > 0:
            raise ValueError(f"Found non-numeric columns after conversion: {non_numeric}")
        
        # Replace any remaining infinities with NaN and then fill with 0
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        return features
        
    except Exception as e:
        logger.error(f"Error in prepare_future_features: {str(e)}")
        raise

def make_predictions(input_data: pd.DataFrame = None, future_periods: int = 52) -> pd.DataFrame:
    """Make predictions using trained models."""
    try:
        # Load configuration and models
        config = load_config()
        model_path = Path(config['output']['model_path'])
        
        if not model_path.exists():
            raise FileNotFoundError(f"No trained models found at {model_path}")
            
        trained_models = joblib.load(model_path)
        logger.info(f"Loaded {len(trained_models)} trained models")
        logger.info(f"Available regions in trained models: {list(trained_models.keys())}")
        
        # Initialize preprocessor
        processor = DataProcessor(config)
        
        # If no input data provided, load from config path
        if input_data is None:
            input_data = pd.read_csv(config['data']['input_path'])
        
        logger.info(f"Input data shape: {input_data.shape}")
        logger.info(f"Columns in input data: {input_data.columns.tolist()}")
        
        # Check if required columns exist
        required_columns = [config['data']['date_column'], 
                          config['data']['target_column'],
                          config['data']['group_by_column']]
        missing_columns = [col for col in required_columns if col not in input_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Get the last date from input data
        last_date = pd.to_datetime(input_data[config['data']['date_column']]).max()
        logger.info(f"Last date in input data: {last_date}")
        
        # Generate future dates
        future_dates = generate_future_dates(last_date, future_periods)
        logger.info(f"Generated {len(future_dates)} future dates")
        
        # Make predictions for each region
        predictions = {}
        for region, model_info in trained_models.items():
            try:
                logger.info(f"\nProcessing region: {region}")
                
                # Get region-specific historical data
                region_data = input_data[input_data[config['data']['group_by_column']] == region].copy()
                logger.info(f"Historical data for {region}: {len(region_data)} rows")
                
                if len(region_data) == 0:
                    logger.warning(f"No historical data found for {region}. Skipping.")
                    continue
                
                # Sort data by date
                region_data = region_data.sort_values(config['data']['date_column'])
                
                # Take the last N rows of historical data to help with feature generation
                lookback_periods = max(processor.config['data']['feature_engineering']['lag_features']) + \
                                 max(processor.config['data']['feature_engineering']['rolling_features']['windows'])
                historical_data = region_data.tail(lookback_periods).copy()
                
                # Combine with future dates
                future_data = future_dates.copy()
                future_data[config['data']['target_column']] = historical_data[config['data']['target_column']].iloc[-1]
                future_data[config['data']['group_by_column']] = region
                
                combined_data = pd.concat([historical_data, future_data], ignore_index=True)
                logger.info(f"Combined data shape: {combined_data.shape}")
                
                # Prepare features for prediction
                future_features = prepare_future_features(combined_data, processor)
                
                if future_features.empty:
                    logger.warning(f"No features generated for {region}. Skipping.")
                    continue
                
                # Take only the future period features
                future_features = future_features.tail(future_periods)
                
                # Ensure all columns are numeric
                non_numeric_cols = future_features.select_dtypes(exclude=['int64', 'float64']).columns
                if len(non_numeric_cols) > 0:
                    logger.error(f"Non-numeric columns found: {non_numeric_cols}")
                    raise ValueError(f"All features must be numeric. Found non-numeric columns: {non_numeric_cols}")
                
                # Make predictions
                model = model_info['model']
                region_predictions = model.predict(future_features)
                
                if len(region_predictions) > 0:
                    predictions[region] = region_predictions
                    logger.info(f"Generated {len(region_predictions)} predictions for {region}")
                    logger.info(f"Prediction range: {region_predictions.min():.2f} to {region_predictions.max():.2f}")
                else:
                    logger.warning(f"No predictions generated for {region}")
                
            except Exception as e:
                logger.error(f"Error making predictions for {region}: {str(e)}")
                logger.error(f"Traceback:", exc_info=True)
                continue
        
        if not predictions:
            raise ValueError("No predictions were generated for any region")
        
        # Combine predictions into a DataFrame
        prediction_df = pd.DataFrame({
            'Week Ending Date': future_dates['Week Ending Date']
        })
        
        for region, preds in predictions.items():
            prediction_df[region] = preds
            
        logger.info(f"\nFinal prediction DataFrame shape: {prediction_df.shape}")
        logger.info(f"Regions with predictions: {list(predictions.keys())}")
        
        # Save predictions
        output_dir = Path('results')
        output_dir.mkdir(exist_ok=True)
        
        prediction_path = output_dir / 'predictions.csv'
        prediction_df.to_csv(prediction_path, index=False)
        logger.info(f"Saved predictions to {prediction_path}")
        
        return prediction_df
        
    except Exception as e:
        logger.error(f"Error in prediction pipeline: {str(e)}")
        logger.error(f"Traceback:", exc_info=True)
        raise

if __name__ == "__main__":
    # Example usage
    predictions = make_predictions(future_periods=52)  # Predict one year ahead
    print("Predictions shape:", predictions.shape)
    print("Predictions preview:\n", predictions.head())