import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, Tuple
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelVisualizer:
    def __init__(self, config: dict):
        self.config = config
        self.results_dir = Path(config['output']['visualization_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for better visualizations
        plt.style.use('seaborn')
        
    def plot_actual_vs_predicted(self, region: str, y_true: pd.Series, y_pred: pd.Series, 
                               dates: pd.Index, timestamps: pd.Series = None):
        """Create actual vs predicted plot for a specific region."""
        try:
            plt.figure(figsize=(15, 6))
            
            # Convert timestamps to datetime if provided
            if timestamps is not None:
                x_values = pd.to_datetime(timestamps.astype('int64') * 1e9)
            else:
                x_values = dates
            
            # Plot with improved styling
            plt.plot(x_values, y_true, label='Actual', color='#2ecc71', linewidth=2, marker='o', markersize=4)
            plt.plot(x_values, y_pred, label='Predicted', color='#e74c3c', linewidth=2, marker='o', markersize=4)
            
            plt.title(f'Actual vs Predicted Values - {region}', fontsize=14, pad=20)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Number of Inpatient Beds', fontsize=12)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Add grid for better readability
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend with better positioning
            plt.legend(loc='upper right', frameon=True, framealpha=0.9)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Save the plot
            save_path = self.results_dir / f'actual_vs_predicted_{region}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
            
            if 'region' in metrics_df.columns:
                metrics_df.set_index('region', inplace=True)
            
            # Create the plot with improved styling
            plt.figure(figsize=(15, 8))
            
            # Plot metrics with different colors and patterns
            colors = ['#2ecc71', '#e74c3c', '#3498db']
            metrics_df[['mape', 'mae', 'rmse']].plot(kind='bar', color=colors)
            
            plt.title('Model Performance Metrics by Region', fontsize=14, pad=20)
            plt.xlabel('Region', fontsize=12)
            plt.ylabel('Metric Value', fontsize=12)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            
            # Add grid for better readability
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Add legend with better positioning
            plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Save plot with high resolution
            plt.savefig(self.results_dir / 'performance_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating performance summary: {str(e)}")
            raise 