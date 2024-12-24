import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from yaml file"""
    try:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        logger.info(f"Config loaded successfully from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise