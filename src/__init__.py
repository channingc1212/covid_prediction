# . means "current directory"
from .predict import make_predictions as predict 
from .train import train_model

__all__ = ['predict', 'train_model']
