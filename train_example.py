"""
Example training script using the refactored modules.

This script demonstrates how to:
1. Load preprocessed data
2. Train multiple models
3. Evaluate performance
4. Save the best model for production
"""

import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path

from src.preprocessing import RobustProductDataPreprocessor
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator


def load_preprocessed_fold(fold_dir: str, data_type: str = 'train'):
    """
    Load preprocessed features and targets from a fold.

    Args:
        fold_dir: Directory containing fold data
        data_type: 'train' or 'val'

    Returns:
        Tuple of (features, targets)
    """
    features_path = os.path.join(fold_dir, f'{data_type}_features.csv')
    processed_path = os.path.join(fold_dir, f'{data_type}_processed.csv')

    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features not found: {features_path}")

    X = pd.read_csv(features_path).values
    processed_df = pd.read_csv(processed_path)
    y = processed_df['price'].values

    print(f"✅ Loaded {data_type}: X={X.shape}, y={y.shape}")
    return X, y


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("PRODUCT PRICE PREDICTION - TRAINING PIPELINE")
    print("=" * 80)

    # Configuration
    PREPROCESSED_DATA_DIR = './preprocessed_data'  # Update with your path
    ARTIFACTS_DIR = './artifacts'
    FOLD_NUM = 0  # Using fold 0 as example

    # Ensure artifacts directory exists
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(ARTIFACTS_DIR, 'evaluation'), exist_ok=True)

    # ============================================================================
    # STEP 1: Load Preprocessed Data
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Loading preprocessed data")
    print("=" * 80)

    fold_dir = os.path.join(PREPROCESSED_DATA_DIR, f'fold_{FOLD_NUM}')

    if not os.path.exists(fold_dir):
        print(f"\n❌ Fold directory not found: {fold_dir}")
        print("\nExpected directory structure:")
        print(f"  {fold_dir}/")
        print(f"    ├── train_features.csv")
        print(f"    ├── train_processed.csv")
        print(f"    ├── val_features.csv")
        print(f"    └── val_processed.csv")
        print("\nIf you have the original notebook's preprocessed data:")
        print("  1. Update PREPROCESSED_DATA_DIR to point to your data")
        print("  2. Ensure the folder structure matches above")
        print("\nAlternatively, run preprocessing on your raw data:")
        print("  from src.preprocessing import RobustProductDataPreprocessor")
        return

    try:
        X_train, y_train = load_preprocessed_fold(fold_dir, 'train')
        X_val, y_val = load_preprocessed_fold(fold_dir, 'val')
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return

    print(f"\n📊 Data Summary:")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Validation samples: {X_val.shape[0]}")
    print(f"   Feature count: {X_train.shape[1]}")
    print(f"   Price range: ${y_train.min():.2f} - ${y_train.max():.2f}")

    # ============================================================================
    # STEP 2: Train Models
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Training models")
    print("=" * 80)

    trainer = ModelTrainer(output_dir=ARTIFACTS_DIR)

    # Train all models (or specify a subset)
    models_to_train = [
        'lightgbm',
        'xgboost',
        'random_forest',
        'gradient_boosting',
        'ridge',
        'lasso'
    ]

    results = trainer.train_all_models(
        X_train, y_train,
        X_val, y_val,
        models_to_train=models_to_train
    )

    # ============================================================================
    # STEP 3: Compare Model Performance
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Model Performance Comparison")
    print("=" * 80)

    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Train Metrics:")
        for metric, value in result['train_metrics'].items():
            print(f"    {metric:10s}: {value:.6f}")

        if 'val_metrics' in result:
            print(f"  Validation Metrics:")
            for metric, value in result['val_metrics'].items():
                print(f"    {metric:10s}: {value:.6f}")

    # ============================================================================
    # STEP 4: Select and Save Best Model
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Selecting best model")
    print("=" * 80)

    best_name, best_model = trainer.get_best_model(metric='smape')
    print(f"\n🏆 Best model: {best_name.upper()}")
    print(f"   Validation SMAPE: {results[best_name]['val_metrics']['smape']:.6f}")

    # Save best model
    model_path = trainer.save_best_model(metric='smape', output_name='model.pkl')
    print(f"✅ Saved to: {model_path}")

    # ============================================================================
    # STEP 5: Save Preprocessor
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Saving preprocessor")
    print("=" * 80)

    # Load the preprocessor that was used in preprocessing
    preprocessor_source = os.path.join(PREPROCESSED_DATA_DIR, 'fitted_preprocessor.pkl')
    if os.path.exists(preprocessor_source):
        preprocessor_path = os.path.join(ARTIFACTS_DIR, 'preprocessor.pkl')
        preprocessor = joblib.load(preprocessor_source)
        joblib.dump(preprocessor, preprocessor_path)
        print(f"✅ Saved preprocessor to: {preprocessor_path}")
    else:
        print(f"⚠️  Preprocessor not found at: {preprocessor_source}")
        print("   The Streamlit app may need the preprocessor to be manually saved")

    # ============================================================================
    # STEP 6: Evaluate Best Model
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 6: Detailed evaluation")
    print("=" * 80)

    evaluator = ModelEvaluator(output_dir=os.path.join(ARTIFACTS_DIR, 'evaluation'))

    # Evaluate on test set (using validation as proxy for this example)
    evaluator.evaluate_and_visualize(
        best_model,
        X_val,
        y_val,
        model_name=best_name
    )

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)

    print(f"\n📁 Artifacts saved to: {ARTIFACTS_DIR}")
    print(f"   ├── model.pkl (Best model)")
    print(f"   ├── preprocessor.pkl (Preprocessor)")
    print(f"   ├── *_model.pkl (Individual models)")
    print(f"   └── evaluation/")
    print(f"       ├── evaluation_metrics.json")
    print(f"       ├── evaluation_report.txt")
    print(f"       └── *.png (Visualizations)")

    print(f"\n🚀 Next steps:")
    print(f"   1. Run Streamlit app: streamlit run app.py")
    print(f"   2. Check evaluation results: artifacts/evaluation/")
    print(f"   3. Deploy model to production")

    return best_model, results


if __name__ == "__main__":
    best_model, results = main()
