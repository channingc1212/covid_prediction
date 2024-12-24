from src.train import train_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        metrics = train_model()
        logger.info("Training pipeline completed successfully")
        logger.info(f"Final metrics: {metrics}")
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 