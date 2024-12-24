import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class ModelVisualizer:
    def __init__(self, config: dict):
        self.config = config
        self.results_dir = Path(config['output']['visualization_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_actual_vs_predicted(self, region: str, y_true: pd.Series, y_pred: pd.Series, dates: pd.Index):
        """Create actual vs predicted plot for a specific region."""
        try:
            plt.figure(figsize=(15, 6))
            plt.plot(dates, y_true, label='Actual', marker='o')
            plt.plot(dates, y_pred, label='Predicted', marker='o')
            plt.title(f'Actual vs Predicted Values - {region}')
            plt.xlabel('Date')
            plt.ylabel('Number of Inpatient Beds')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            
            # Save the plot
            save_path = self.results_dir / f'actual_vs_predicted_{region}.png'
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Saved actual vs predicted plot for {region} at {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating actual vs predicted plot for {region}: {str(e)}")
            
    def plot_error_distribution(self, group: str, target: pd.Series, predictions: np.ndarray):
        """Plot error distribution for a group"""
        errors = target - predictions
        
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.title(f'Prediction Error Distribution - {group}')
        plt.xlabel('Error')
        plt.ylabel('Count')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.results_dir / f'error_distribution_{group}.png')
        plt.close()
        
    def create_performance_summary(self, region_metrics):
        """Create a summary visualization of model performance across regions."""
        try:
            # Convert metrics dictionary to DataFrame
            metrics_data = []
            for region, metrics in region_metrics.items():
                metrics_dict = metrics.copy()
                metrics_dict['region'] = region
                metrics_data.append(metrics_dict)
            
            metrics_df = pd.DataFrame(metrics_data)
            
            # Debug logging
            logger.info(f"Metrics DataFrame columns: {metrics_df.columns.tolist()}")
            logger.info(f"Metrics DataFrame head:\n{metrics_df.head()}")
            
            # Set region as index
            if 'region' in metrics_df.columns:
                metrics_df.set_index('region', inplace=True)
            
            # Create the plot
            plt.figure(figsize=(15, 8))
            metrics_df[['mae', 'rmse', 'mape']].plot(kind='bar')
            plt.title('Model Performance Metrics by Region')
            plt.xlabel('Region')
            plt.ylabel('Metric Value')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.results_dir / 'performance_summary.png')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating performance summary: {str(e)}")
            raise 