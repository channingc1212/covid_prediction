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
    """Class for creating visualizations of model results."""
    
    def __init__(self, config):
        """Initialize the visualizer with configuration."""
        self.config = config
        
        # Set default visualization settings if not in config
        self.viz_config = config.get('visualizations', {
            'plot_style': 'seaborn-v0_8',
            'figure_size': [12, 6],
            'save_format': 'png'
        })
        
        # Create visualization directory
        self.viz_dir = Path(config['output']['visualization_dir'])
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Use the style from config
        plt.style.use(self.viz_config['plot_style'])
        plt.rcParams['figure.figsize'] = self.viz_config['figure_size']
        
    def plot_actual_vs_predicted(self, region: str, y_true: pd.Series, y_pred: pd.Series, 
                               dates: pd.Index, timestamps: pd.Series = None):
        """Create actual vs predicted plot for a specific region."""
        try:
            # Debug information
            logger.info(f"Creating plot for region {region}")
            logger.info(f"Data shapes - y_true: {y_true.shape}, y_pred: {y_pred.shape}, dates: {len(dates)}")
            
            # Create figure with specified size
            plt.figure(figsize=(15, 8))
            
            # Convert timestamps to datetime if provided
            if timestamps is not None:
                logger.info("Using provided timestamps")
                try:
                    x_values = pd.to_datetime(timestamps.astype('int64') * 1e9)
                except Exception as e:
                    logger.error(f"Error converting timestamps: {str(e)}")
                    x_values = pd.to_datetime(dates)
            else:
                logger.info("Using dates index")
                x_values = pd.to_datetime(dates)
            
            logger.info(f"Date range: {x_values.min()} to {x_values.max()}")
            
            # Create a DataFrame with proper alignment
            try:
                plot_df = pd.DataFrame({
                    'date': x_values,
                    'actual': y_true.values,  # Convert Series to array
                    'predicted': y_pred  # Already numpy array
                })
                
                # Sort by date
                plot_df = plot_df.sort_values('date')
                logger.info(f"Plot DataFrame shape: {plot_df.shape}")
                
            except Exception as e:
                logger.error(f"Error creating plot DataFrame: {str(e)}")
                raise
            
            # Plot with clean styling
            plt.plot(plot_df['date'], plot_df['actual'], label='Actual', 
                    color='#2ecc71', linewidth=2, alpha=0.8)
            plt.plot(plot_df['date'], plot_df['predicted'], label='Predicted',
                    color='#e74c3c', linewidth=2, alpha=0.8,
                    linestyle='--')
            
            # Customize the plot
            plt.title(f'Actual vs Predicted Values - {region}', 
                     fontsize=14, pad=20)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Number of Inpatient Beds', fontsize=12)
            
            # Format x-axis
            try:
                plt.gca().xaxis.set_major_locator(plt.YearLocator())
                plt.gca().xaxis.set_major_formatter(plt.DateFormatter('%Y-%m'))
            except Exception as e:
                logger.error(f"Error formatting x-axis: {str(e)}")
                # Fallback to auto formatting
                plt.gcf().autofmt_xdate()
            
            # Rotate and align the tick labels so they look better
            plt.xticks(rotation=45, ha='right')
            
            # Add subtle grid
            plt.grid(True, linestyle='--', alpha=0.3)
            
            # Add legend with better positioning
            plt.legend(loc='upper right', frameon=True, framealpha=0.9,
                      fontsize=10)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Save the plot
            save_path = self.viz_dir / f'actual_vs_predicted_{region}.{self.viz_config["save_format"]}'
            logger.info(f"Attempting to save plot to {save_path}")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Successfully saved plot for {region}")
            
        except Exception as e:
            logger.error(f"Error creating actual vs predicted plot for {region}")
            logger.error(f"Error details: {str(e)}")
            logger.error(f"Data types - y_true: {type(y_true)}, y_pred: {type(y_pred)}, dates: {type(dates)}")
            plt.close()  # Ensure figure is closed even if error occurs
            
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
        plt.savefig(self.viz_dir / f'error_distribution_{group}.{self.viz_config["save_format"]}')
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
            plt.grid(True, axis='y', linestyle='--', alpha=0.3)
            
            # Add legend with better positioning
            plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Save plot with high resolution
            plt.savefig(self.viz_dir / f'performance_summary.{self.viz_config["save_format"]}',
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating performance summary: {str(e)}")
            raise 