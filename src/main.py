import logging
from src.train import train_model
from src.utils import load_config
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        'models',
        'results',
        'results/visualizations',
        'visualizations',
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def main():
    try:
        # Create directories
        setup_directories()
        
        # Train models
        logger.info("Starting model training...")
        models = train_model()
        
        logger.info("Training completed successfully!")
        return models
        
    except KeyError as e:
        logger.error(f"Configuration error: Missing key {str(e)}")
        logger.info("Please check your config.yaml file for missing configurations")
        raise
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 