"""Feature extraction utilities for images and advanced text processing."""

import numpy as np
from typing import List, Optional
import warnings

warnings.filterwarnings('ignore')


class ImageFeatureExtractor:
    """
    Image feature extraction using ResNet50.

    This class provides utilities for extracting deep learning features
    from product images. It requires torch and torchvision.
    """

    def __init__(self, model_name: str = 'resnet50', feature_dim: int = 2048):
        """
        Initialize the image feature extractor.

        Args:
            model_name: Pre-trained model to use ('resnet50', etc.)
            feature_dim: Dimension of extracted features

        Note:
            This is a placeholder for the full OptimizedImageFeatureExtractor
            from the original notebook. Use this for reference and future enhancement.
        """
        self.model_name = model_name
        self.feature_dim = feature_dim

    def extract_features(self, image_urls: List[str]) -> np.ndarray:
        """
        Extract features from a list of image URLs.

        Args:
            image_urls: List of image URLs

        Returns:
            Array of shape (n_images, feature_dim)

        Note:
            Full implementation available in the original notebook's
            OptimizedImageFeatureExtractor class.
        """
        raise NotImplementedError(
            "Full image feature extraction requires the OptimizedImageFeatureExtractor "
            "from the original notebook. See notebooks/main.ipynb for implementation."
        )


def combine_text_and_image_features(
    text_features: np.ndarray,
    image_features: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Combine text and image features.

    Args:
        text_features: Text-derived features
        image_features: Image-derived features (optional)

    Returns:
        Combined feature array

    Note:
        This function handles alignment and combination of multi-modal features.
    """
    if image_features is None:
        return text_features

    if text_features.shape[0] != image_features.shape[0]:
        raise ValueError(
            f"Mismatched sample counts: text={text_features.shape[0]}, "
            f"image={image_features.shape[0]}"
        )

    return np.hstack([text_features, image_features])
