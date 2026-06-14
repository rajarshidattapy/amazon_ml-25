"""Production inference pipeline for model predictions."""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import os
from .preprocessing import RobustProductDataPreprocessor


class ProductPricePredictor:
    """Production-ready predictor for product prices."""

    def __init__(self, model_path: str, preprocessor_path: str, artifacts_dir: str = "./artifacts"):
        """
        Initialize the predictor.

        Args:
            model_path: Path to saved model
            preprocessor_path: Path to saved preprocessor
            artifacts_dir: Base artifacts directory
        """
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.artifacts_dir = artifacts_dir

    def validate_input(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate input data.

        Args:
            data: Input dictionary with catalog_content

        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ['catalog_content']

        for field in required_fields:
            if field not in data:
                return False, f"Missing required field: {field}"

            if not isinstance(data[field], str) or len(data[field].strip()) == 0:
                return False, f"Field '{field}' must be a non-empty string"

        return True, ""

    def preprocess_input(self, data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess raw input into features.

        Args:
            data: Raw input dictionary

        Returns:
            Preprocessed feature dataframe
        """
        # Create a simple dataframe with required structure
        df = pd.DataFrame([{
            'catalog_content': data['catalog_content'],
            'price': 0.0  # Placeholder, not used in inference
        }])

        features, processed = self.preprocessor.preprocess_data_robust(df, is_training=False)
        return features

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction for a single product.

        Args:
            data: Dictionary with 'catalog_content' key

        Returns:
            Dictionary with prediction and metadata
        """
        # Validate input
        is_valid, error_msg = self.validate_input(data)
        if not is_valid:
            return {
                'success': False,
                'error': error_msg,
                'prediction': None
            }

        try:
            # Preprocess input
            features = self.preprocess_input(data)

            # Make prediction
            prediction = self.model.predict(features.values)[0]

            # Ensure non-negative price
            prediction = max(0.0, float(prediction))

            return {
                'success': True,
                'prediction': prediction,
                'error': None,
                'feature_count': features.shape[1]
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'prediction': None
            }

    def predict_batch(self, data_list: list) -> list:
        """
        Make predictions for multiple products.

        Args:
            data_list: List of input dictionaries

        Returns:
            List of prediction results
        """
        results = []
        for data in data_list:
            results.append(self.predict(data))
        return results
