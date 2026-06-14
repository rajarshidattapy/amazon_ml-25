"""Evaluation pipeline for model performance assessment."""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .utils import smape, calculate_regression_metrics


class ModelEvaluator:
    """Comprehensive model evaluator with metrics and visualizations."""

    def __init__(self, output_dir: str = "./artifacts/evaluation"):
        """
        Initialize evaluator.

        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {}

    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model

        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = calculate_regression_metrics(y_test, y_pred)
        metrics['model_name'] = model_name

        self.results[model_name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'y_test': y_test
        }

        return metrics

    def save_metrics(self, filename: str = "evaluation_metrics.json"):
        """
        Save evaluation metrics to JSON.

        Args:
            filename: Output filename
        """
        metrics_dict = {}
        for model_name, data in self.results.items():
            metrics_dict[model_name] = data['metrics']

        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        print(f"✅ Metrics saved to {filepath}")

    def plot_predictions_vs_actual(self, model_name: str = None, figsize: Tuple = (10, 6)):
        """
        Create predictions vs actual plot.

        Args:
            model_name: Which model to plot (None = first model)
            figsize: Figure size
        """
        if not self.results:
            print("No results to plot")
            return

        if model_name is None:
            model_name = list(self.results.keys())[0]

        y_test = self.results[model_name]['y_test']
        y_pred = self.results[model_name]['y_pred']

        plt.figure(figsize=figsize)
        plt.scatter(y_test, y_pred, alpha=0.5, edgecolors='k')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title(f'{model_name}: Predictions vs Actual')
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, f'{model_name}_predictions_vs_actual.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Prediction plot saved to {filepath}")

    def plot_residuals(self, model_name: str = None, figsize: Tuple = (10, 6)):
        """
        Create residuals plot.

        Args:
            model_name: Which model to plot (None = first model)
            figsize: Figure size
        """
        if not self.results:
            print("No results to plot")
            return

        if model_name is None:
            model_name = list(self.results.keys())[0]

        y_test = self.results[model_name]['y_test']
        y_pred = self.results[model_name]['y_pred']
        residuals = y_test - y_pred

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Residuals vs predicted
        axes[0].scatter(y_pred, residuals, alpha=0.5, edgecolors='k')
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Price')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')

        # Residuals distribution
        axes[1].hist(residuals, bins=30, edgecolor='k', alpha=0.7)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Residuals')

        plt.tight_layout()

        filepath = os.path.join(self.output_dir, f'{model_name}_residuals.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Residuals plot saved to {filepath}")

    def plot_error_distribution(self, model_name: str = None, figsize: Tuple = (10, 6)):
        """
        Create error distribution plot.

        Args:
            model_name: Which model to plot (None = first model)
            figsize: Figure size
        """
        if not self.results:
            print("No results to plot")
            return

        if model_name is None:
            model_name = list(self.results.keys())[0]

        y_test = self.results[model_name]['y_test']
        y_pred = self.results[model_name]['y_pred']
        errors = np.abs(y_test - y_pred)

        plt.figure(figsize=figsize)
        plt.hist(errors, bins=50, edgecolor='k', alpha=0.7)
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.title(f'{model_name}: Distribution of Absolute Errors')
        plt.axvline(errors.mean(), color='r', linestyle='--', lw=2, label=f'Mean: {errors.mean():.2f}')
        plt.legend()
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, f'{model_name}_error_distribution.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Error distribution plot saved to {filepath}")

    def plot_metrics_comparison(self, figsize: Tuple = (12, 6)):
        """
        Create metrics comparison plot across all models.

        Args:
            figsize: Figure size
        """
        if not self.results:
            print("No results to compare")
            return

        metrics_list = []
        for model_name, data in self.results.items():
            metrics = data['metrics'].copy()
            metrics_list.append(metrics)

        df_metrics = pd.DataFrame(metrics_list)

        # Select numeric metrics for comparison
        metric_cols = ['mae', 'rmse', 'r2', 'smape']
        metric_cols = [col for col in metric_cols if col in df_metrics.columns]

        fig, axes = plt.subplots(1, len(metric_cols), figsize=figsize)
        if len(metric_cols) == 1:
            axes = [axes]

        for idx, metric in enumerate(metric_cols):
            df_metrics.set_index('model_name')[metric].plot(kind='bar', ax=axes[idx])
            axes[idx].set_title(f'{metric.upper()}')
            axes[idx].set_ylabel('Score')
            axes[idx].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        filepath = os.path.join(self.output_dir, 'metrics_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Metrics comparison plot saved to {filepath}")

    def generate_report(self, filename: str = "evaluation_report.txt"):
        """
        Generate comprehensive evaluation report.

        Args:
            filename: Output filename
        """
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            for model_name, data in self.results.items():
                f.write(f"\n{model_name.upper()}\n")
                f.write("-" * 40 + "\n")

                metrics = data['metrics']
                for metric_name, value in metrics.items():
                    if metric_name != 'model_name':
                        f.write(f"{metric_name.upper():15s}: {value:.6f}\n")

        print(f"✅ Report saved to {filepath}")

    def evaluate_and_visualize(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model"
    ):
        """
        Evaluate model and create all visualizations.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
        """
        # Evaluate
        metrics = self.evaluate_model(model, X_test, y_test, model_name)

        # Save metrics
        self.save_metrics()

        # Create visualizations
        self.plot_predictions_vs_actual(model_name)
        self.plot_residuals(model_name)
        self.plot_error_distribution(model_name)
        self.plot_metrics_comparison()

        # Generate report
        self.generate_report()

        print(f"\n✅ Complete evaluation for {model_name}:")
        for metric, value in metrics.items():
            if metric != 'model_name':
                print(f"   {metric.upper()}: {value:.6f}")
