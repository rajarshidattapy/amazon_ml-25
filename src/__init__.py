"""ML pipeline package for product price prediction."""

from .utils import smape, calculate_regression_metrics
from .preprocessing import RobustProductDataPreprocessor
from .features import ImageFeatureExtractor, combine_text_and_image_features
from .train import ModelTrainer
from .evaluate import ModelEvaluator
from .inference import ProductPricePredictor

__all__ = [
    'smape',
    'calculate_regression_metrics',
    'RobustProductDataPreprocessor',
    'ImageFeatureExtractor',
    'combine_text_and_image_features',
    'ModelTrainer',
    'ModelEvaluator',
    'ProductPricePredictor',
]

__version__ = '1.0.0'
