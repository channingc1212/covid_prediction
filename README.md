# COVID-19 Hospital Bed Prediction

A machine learning pipeline for predicting COVID-19 hospital bed utilization across different regions.

## Project Structure

```
covid_prediction/
├── config/                  # Configuration files
├── data/                   # Data directory
├── models/                 # Saved models
├── results/                # Results and visualizations
├── logs/                   # Logging output
├── src/                    # Source code
├── tests/                  # Test files
└── requirements.txt        # Dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the models for all regions:
```bash
python -m src.main
```

### Making Predictions

Generate predictions for future periods:
```bash
python run_prediction.py --periods 52  # Predict one year ahead
```

Optional arguments:
- `--periods`: Number of weeks to predict (default: 52)
- `--input-file`: Path to custom input data file

### Output

- Trained models are saved in `models/`
- Predictions are saved in `results/predictions.csv`
- Visualizations are saved in `results/visualizations/`
- Logs are saved in `logs/prediction.log`

## Configuration

The `config/config.yaml` file contains all configurable parameters:
- Data paths and column names
- Feature engineering settings
- Model parameters
- Output paths

## Features

- Multi-region prediction support
- Time series feature engineering
- Multiple model support (Random Forest, XGBoost)
- Automated feature generation
- Visualization generation
- Comprehensive logging

## Data Format

Input data should include:
- Date column (weekly)
- Target column (number of inpatient beds)
- Region identifier
- Additional features for prediction

## Model Pipeline

1. Data preprocessing
   - Time feature generation
   - Lag feature creation
   - Rolling statistics
   - Missing value handling

2. Model training
   - Per-region model training
   - Model selection based on MAPE
   - Performance evaluation

3. Prediction
   - Future date generation
   - Feature preparation
   - Multi-region prediction
   - Results aggregation

## Visualization

The pipeline generates:
- Actual vs predicted plots for each region
- Performance metrics comparison
- Error distribution plots

## Logging

Detailed logs are saved in `logs/prediction.log`, including:
- Data processing steps
- Feature generation details
- Model performance metrics
- Error messages and warnings

## Testing

Run integration tests:
```bash
python -m pytest tests/test_pipeline.py
``` 