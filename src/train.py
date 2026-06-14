"""Model training module with support for multiple algorithms."""

import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
import xgboost as xgb
import lightgbm as lgb

from .utils import calculate_regression_metrics, smape


class ModelTrainer:
    """Trainer for multiple regression models."""

    def __init__(self, output_dir: str = "./artifacts"):
        """
        Initialize trainer.

        Args:
            output_dir: Directory to save trained models
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.models = {}
        self.results = {}

    def train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs
    ) -> lgb.LGBMRegressor:
        """Train LightGBM model."""
        params = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        params.update(kwargs)

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)

        return model

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs
    ) -> xgb.XGBRegressor:
        """Train XGBoost model."""
        params = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 5,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 0
        }
        params.update(kwargs)

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        return model

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs
    ) -> RandomForestRegressor:
        """Train Random Forest model."""
        params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 10,
            'random_state': 42,
            'n_jobs': -1
        }
        params.update(kwargs)

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        return model

    def train_gradient_boosting(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs
    ) -> GradientBoostingRegressor:
        """Train Gradient Boosting model."""
        params = {
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 5,
            'random_state': 42
        }
        params.update(kwargs)

        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)

        return model

    def train_ridge(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs
    ) -> Ridge:
        """Train Ridge Regression model."""
        params = {
            'alpha': 1.0,
            'random_state': 42
        }
        params.update(kwargs)

        model = Ridge(**params)
        model.fit(X_train, y_train)

        return model

    def train_lasso(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs
    ) -> Lasso:
        """Train Lasso Regression model."""
        params = {
            'alpha': 0.001,
            'random_state': 42,
            'max_iter': 10000
        }
        params.update(kwargs)

        model = Lasso(**params)
        model.fit(X_train, y_train)

        return model

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        models_to_train: list = None
    ) -> Dict[str, Any]:
        """
        Train multiple models.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            models_to_train: List of model names to train (None = all)

        Returns:
            Dictionary with trained models and results
        """
        if models_to_train is None:
            models_to_train = [
                'lightgbm', 'xgboost', 'random_forest',
                'gradient_boosting', 'ridge', 'lasso'
            ]

        trainers = {
            'lightgbm': self.train_lightgbm,
            'xgboost': self.train_xgboost,
            'random_forest': self.train_random_forest,
            'gradient_boosting': self.train_gradient_boosting,
            'ridge': self.train_ridge,
            'lasso': self.train_lasso
        }

        results = {}

        for model_name in models_to_train:
            if model_name not in trainers:
                print(f"⚠️  Unknown model: {model_name}")
                continue

            print(f"\n🚀 Training {model_name.upper()}...")
            trainer_fn = trainers[model_name]

            # Train model
            model = trainer_fn(X_train, y_train, X_val, y_val)
            self.models[model_name] = model

            # Evaluate on training data
            y_train_pred = model.predict(X_train)
            train_metrics = calculate_regression_metrics(y_train, y_train_pred)

            results[model_name] = {
                'model': model,
                'train_metrics': train_metrics
            }

            # Evaluate on validation data if provided
            if X_val is not None and y_val is not None:
                y_val_pred = model.predict(X_val)
                val_metrics = calculate_regression_metrics(y_val, y_val_pred)
                results[model_name]['val_metrics'] = val_metrics

                print(f"   Train SMAPE: {train_metrics['smape']:.4f}")
                print(f"   Val SMAPE:   {val_metrics['smape']:.4f}")
                print(f"   Val R²:      {val_metrics['r2']:.4f}")
            else:
                print(f"   Train SMAPE: {train_metrics['smape']:.4f}")

            # Save model
            model_path = os.path.join(self.output_dir, f'{model_name}_model.pkl')
            joblib.dump(model, model_path)
            print(f"   ✅ Saved to {model_path}")

        self.results = results
        return results

    def get_best_model(self, metric: str = 'smape', dataset: str = 'val') -> Tuple[str, Any]:
        """
        Get best performing model.

        Args:
            metric: Metric to optimize (default: smape)
            dataset: Dataset to use ('train' or 'val')

        Returns:
            Tuple of (model_name, model)
        """
        if dataset == 'val':
            metric_key = 'val_metrics'
        else:
            metric_key = 'train_metrics'

        best_model_name = None
        best_value = float('inf') if metric in ['mae', 'mse', 'rmse', 'smape'] else float('-inf')

        for model_name, result in self.results.items():
            if metric_key not in result:
                continue

            value = result[metric_key].get(metric)
            if value is None:
                continue

            if metric in ['mae', 'mse', 'rmse', 'smape']:
                if value < best_value:
                    best_value = value
                    best_model_name = model_name
            else:  # r2, etc - higher is better
                if value > best_value:
                    best_value = value
                    best_model_name = model_name

        if best_model_name is None:
            return None, None

        return best_model_name, self.models[best_model_name]

    def save_best_model(self, metric: str = 'smape', output_name: str = 'best_model.pkl') -> str:
        """
        Save best model to artifacts.

        Args:
            metric: Metric to optimize
            output_name: Output filename

        Returns:
            Path to saved model
        """
        model_name, best_model = self.get_best_model(metric)

        if best_model is None:
            print("❌ No model to save")
            return None

        output_path = os.path.join(self.output_dir, output_name)
        joblib.dump(best_model, output_path)
        print(f"✅ Best model ({model_name}) saved to {output_path}")

        return output_path
