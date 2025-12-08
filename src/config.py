"""
Configuration settings for the Antibiotic Resistance Analysis System
"""

from pathlib import Path
from typing import Dict, Any
import json
import logging

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration manager for the analysis pipeline
    """
    
    # Project paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / 'data'
    RAW_DATA_DIR = DATA_DIR / 'raw'
    PROCESSED_DATA_DIR = DATA_DIR / 'processed'
    RESULTS_DIR = DATA_DIR / 'results'
    MODELS_DIR = DATA_DIR / 'models'
    
    # File names
    RAW_DATA_FILE = 'raw_data.csv'
    CLEANED_DATA_FILE = 'cleaned_data.csv'
    FEATURES_FILE = 'features.csv'
    
    # Data split ratios
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Classification settings
    CLASSIFICATION_TASKS = ['species', 'mar_class', 'susceptibility']
    RANDOM_STATE = 42
    N_JOBS = -1  # Use all available cores
    
    # Model hyperparameters
    MODEL_PARAMS = {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': -1
        },
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1
        },
        'neural_network': {
            'hidden_layer_sizes': (100, 50),
            'max_iter': 500,
            'random_state': 42
        }
    }
    
    # Clustering settings
    CLUSTERING_METHODS = ['kmeans', 'hierarchical', 'dbscan']
    KMEANS_RANGE = (2, 10)  # Range of k values to try
    DBSCAN_EPS = 0.5
    DBSCAN_MIN_SAMPLES = 5
    
    # Association rules settings
    MIN_SUPPORT = 0.1
    MIN_CONFIDENCE = 0.7
    MIN_LIFT = 1.2
    
    # Dimensionality reduction settings
    PCA_N_COMPONENTS = 2
    TSNE_N_COMPONENTS = 2
    TSNE_PERPLEXITY = 30
    UMAP_N_COMPONENTS = 2
    UMAP_N_NEIGHBORS = 15
    
    # MIC transformation settings
    MIN_MIC_VALUE = 0.001  # Minimum value to avoid log(0)
    DEFAULT_LOG_MIC = 0    # Default value when all MIC values are invalid
    
    # Visualization settings
    FIGURE_SIZE = (12, 8)
    DPI = 300
    PLOT_STYLE = 'seaborn-v0_8-darkgrid'
    COLOR_PALETTE = 'Set2'
    
    # Logging settings
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = BASE_DIR / 'pipeline.log'
    
    # Performance settings
    ENABLE_CACHING = True
    CACHE_DIR = BASE_DIR / '.cache'
    PARALLEL_PROCESSING = True
    MAX_WORKERS = None  # None = auto-detect
    
    # Export settings
    EXPORT_FORMATS = ['csv', 'json', 'excel']
    EXPORT_DIR = RESULTS_DIR / 'exports'
    
    @classmethod
    def load_from_file(cls, config_file: Path) -> None:
        """
        Load configuration from JSON file
        
        Args:
            config_file: Path to configuration JSON file
        """
        try:
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
            
            for key, value in config_dict.items():
                if hasattr(cls, key):
                    setattr(cls, key, value)
                    logger.info(f"Loaded config: {key} = {value}")
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_file}")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    
    @classmethod
    def save_to_file(cls, config_file: Path) -> None:
        """
        Save current configuration to JSON file
        
        Args:
            config_file: Path to save configuration JSON file
        """
        config_dict = {}
        for key in dir(cls):
            if not key.startswith('_') and key.isupper():
                value = getattr(cls, key)
                # Convert Path objects to strings
                if isinstance(value, Path):
                    value = str(value)
                # Skip methods
                if not callable(value):
                    config_dict[key] = value
        
        try:
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            logger.info(f"Saved config to: {config_file}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    @classmethod
    def create_directories(cls) -> None:
        """Create all required directories if they don't exist"""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.RESULTS_DIR,
            cls.MODELS_DIR,
            cls.CACHE_DIR,
            cls.EXPORT_DIR,
            cls.RESULTS_DIR / 'clustering',
            cls.RESULTS_DIR / 'association_rules',
            cls.RESULTS_DIR / 'dimensionality_reduction',
            cls.RESULTS_DIR / 'statistical_analysis'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    @classmethod
    def get_summary(cls) -> Dict[str, Any]:
        """
        Get configuration summary
        
        Returns:
            Dictionary with configuration settings
        """
        return {
            'Paths': {
                'Base Directory': str(cls.BASE_DIR),
                'Data Directory': str(cls.DATA_DIR),
                'Results Directory': str(cls.RESULTS_DIR)
            },
            'Data Split': {
                'Train': f"{cls.TRAIN_RATIO*100:.0f}%",
                'Validation': f"{cls.VAL_RATIO*100:.0f}%",
                'Test': f"{cls.TEST_RATIO*100:.0f}%"
            },
            'Classification': {
                'Tasks': cls.CLASSIFICATION_TASKS,
                'Random State': cls.RANDOM_STATE
            },
            'Performance': {
                'Caching Enabled': cls.ENABLE_CACHING,
                'Parallel Processing': cls.PARALLEL_PROCESSING,
                'Max Workers': cls.MAX_WORKERS or 'Auto'
            }
        }


# Initialize directories on import
Config.create_directories()
