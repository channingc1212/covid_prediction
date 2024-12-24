from src import train_model, predict
import logging

# logging setup
logging.basicConfig(level=logging.INFO) # set logging level to INFO
logger = logging.getLogger(__name__) # get logger for the current module

# main function to train or predict
def main(mode="train"): # default is "train"
    try:
        if mode == "train":
            metrics = train_model() # import from src.train
            logger.info("Training pipeline completed successfully") # log training completion
            logger.info(f"Final metrics: {metrics}") # log final metrics
        elif mode == "predict":
            # Example input - modify according to your model's requirements
            sample_input = {
                "feature1": 1.0,
                "feature2": 2.0,
                # Add other features as needed
            }
            prediction = predict(sample_input)
            logger.info(f"Prediction result: {prediction}")
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "train" # default to train
    main(mode) 