import argparse
import logging
from pathlib import Path
from src.predict import make_predictions

# Create logs directory if it doesn't exist
Path('logs').mkdir(exist_ok=True)

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/prediction.log'),
        logging.StreamHandler()  # This will still print to console
    ]
)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Generate future predictions for COVID-19 hospital data')
    parser.add_argument('--periods', type=int, default=52,
                      help='Number of weeks to predict into the future (default: 52)')
    parser.add_argument('--input-file', type=str,
                      help='Optional: Path to input data file (default: uses config path)')
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting prediction pipeline...")
        logger.info(f"Predicting {args.periods} weeks into the future")
        
        if args.input_file:
            import pandas as pd
            logger.info(f"Using custom input file: {args.input_file}")
            input_data = pd.read_csv(args.input_file)
            predictions = make_predictions(input_data=input_data, future_periods=args.periods)
        else:
            logger.info("Using default input file from config")
            predictions = make_predictions(future_periods=args.periods)
            
        print("\nPrediction Summary:")
        print(f"Generated predictions for {predictions.shape[1]-1} regions")
        print(f"Time range: {predictions['Week Ending Date'].min()} to {predictions['Week Ending Date'].max()}")
        print("\nPreview of predictions:")
        print(predictions.head())
        print("\nPredictions saved to results/predictions.csv")
        print("\nFull logs available in logs/prediction.log")
        
    except Exception as e:
        logger.error(f"Error running predictions: {str(e)}")
        raise

if __name__ == "__main__":
    main() 