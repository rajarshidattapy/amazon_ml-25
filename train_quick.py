"""
Quick training script for demonstration and testing.
Generates sample data and trains models for the Streamlit app.
"""

import numpy as np
import pandas as pd
import joblib
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

from src.preprocessing import RobustProductDataPreprocessor
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator


def generate_sample_data(n_samples: int = 500):
    """Generate sample product data for demonstration."""
    print(f"\n📊 Generating {n_samples} sample product records...")
    
    np.random.seed(42)
    
    # Create sample data
    products = []
    for i in range(n_samples):
        price = np.random.lognormal(mean=4.5, sigma=1.2)  # Realistic price distribution
        
        product = {
            'price': price,
            'catalog_content': f"""
Item Name: Product {i}
Bullet Point 1: Premium quality material
Bullet Point 2: Durable and long-lasting
Bullet Point 3: Eco-friendly manufacturing
Product Description: This is a high-quality product designed for everyday use. 
It features premium materials and exceptional craftsmanship.
Value: {np.random.randint(100, 1000)}
Unit: {'pieces' if i % 2 == 0 else 'grams'}
            """.strip()
        }
        products.append(product)
    
    df = pd.DataFrame(products)
    print(f"✅ Generated {len(df)} samples with prices ranging ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    return df


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("PRODUCT PRICE PREDICTION - QUICK TRAINING")
    print("=" * 80)

    # Configuration
    ARTIFACTS_DIR = './artifacts'
    
    # Ensure artifacts directory exists
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(ARTIFACTS_DIR, 'evaluation'), exist_ok=True)

    # ============================================================================
    # STEP 1: Generate Sample Data
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 1: Preparing data")
    print("=" * 80)

    df = generate_sample_data(n_samples=500)
    
    # Split into train/val
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"   Training samples: {len(train_df)}")
    print(f"   Validation samples: {len(val_df)}")

    # ============================================================================
    # STEP 2: Preprocess Data
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 2: Preprocessing data")
    print("=" * 80)

    preprocessor = RobustProductDataPreprocessor()
    
    print("   Preprocessing training data...")
    X_train, processed_train = preprocessor.preprocess_data_robust(train_df, is_training=True)
    y_train = processed_train['price'].values
    
    print("   Preprocessing validation data...")
    X_val, processed_val = preprocessor.preprocess_data_robust(val_df, is_training=False)
    y_val = processed_val['price'].values
    
    print(f"\n   ✅ Training features: {X_train.shape}")
    print(f"   ✅ Validation features: {X_val.shape}")

    # Save preprocessor
    preprocessor_path = os.path.join(ARTIFACTS_DIR, 'preprocessor.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    print(f"   ✅ Preprocessor saved to: {preprocessor_path}")

    # ============================================================================
    # STEP 3: Train Models
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 3: Training models")
    print("=" * 80)

    trainer = ModelTrainer(output_dir=ARTIFACTS_DIR)

    models_to_train = [
        'lightgbm',
        'xgboost', 
        'random_forest',
        'ridge'
    ]

    results = trainer.train_all_models(
        X_train, y_train,
        X_val, y_val,
        models_to_train=models_to_train
    )

    # ============================================================================
    # STEP 4: Compare Model Performance
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 4: Model Performance Summary")
    print("=" * 80)

    comparison_data = []
    for model_name, result in results.items():
        train_smape = result['train_metrics']['smape']
        val_smape = result['val_metrics']['smape'] if 'val_metrics' in result else 0
        val_r2 = result['val_metrics']['r2'] if 'val_metrics' in result else 0
        
        comparison_data.append({
            'Model': model_name.upper(),
            'Train SMAPE': f"{train_smape:.4f}",
            'Val SMAPE': f"{val_smape:.4f}",
            'Val R²': f"{val_r2:.4f}"
        })
        
        print(f"\n   {model_name.upper()}:")
        print(f"      Train SMAPE: {train_smape:.4f}")
        print(f"      Val SMAPE:   {val_smape:.4f}")
        print(f"      Val R²:      {val_r2:.4f}")

    # ============================================================================
    # STEP 5: Select Best Model
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 5: Selecting best model")
    print("=" * 80)

    best_name, best_model = trainer.get_best_model(metric='smape', dataset='val')
    
    if best_model is not None:
        print(f"\n   🏆 Best model: {best_name.upper()}")
        val_smape = results[best_name]['val_metrics']['smape']
        val_r2 = results[best_name]['val_metrics']['r2']
        print(f"      Validation SMAPE: {val_smape:.4f}")
        print(f"      Validation R²: {val_r2:.4f}")

        # Save best model
        model_path = trainer.save_best_model(metric='smape', output_name='model.pkl')
        print(f"\n   ✅ Best model saved to: {model_path}")
    else:
        print("   ❌ No models trained successfully")
        return None, results

    # ============================================================================
    # STEP 6: Evaluate Best Model
    # ============================================================================
    print("\n" + "=" * 80)
    print("STEP 6: Generating evaluation report")
    print("=" * 80)

    try:
        evaluator = ModelEvaluator(output_dir=os.path.join(ARTIFACTS_DIR, 'evaluation'))
        evaluator.evaluate_and_visualize(
            best_model,
            X_val,
            y_val,
            model_name=best_name
        )
        print(f"   ✅ Evaluation saved to: {os.path.join(ARTIFACTS_DIR, 'evaluation/')}")
    except Exception as e:
        print(f"   ⚠️  Evaluation error: {e}")

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)

    print(f"\n📁 Models saved:")
    print(f"   ✅ {os.path.join(ARTIFACTS_DIR, 'model.pkl')} (Best model - for Streamlit)")
    print(f"   ✅ {os.path.join(ARTIFACTS_DIR, 'preprocessor.pkl')} (For Streamlit)")
    
    individual_models = [f for f in os.listdir(ARTIFACTS_DIR) if f.endswith('_model.pkl')]
    for model_file in individual_models:
        print(f"   ✓ {os.path.join(ARTIFACTS_DIR, model_file)}")

    print(f"\n📊 Evaluation results:")
    eval_dir = os.path.join(ARTIFACTS_DIR, 'evaluation')
    if os.path.exists(eval_dir):
        eval_files = os.listdir(eval_dir)
        for file in sorted(eval_files):
            print(f"   ✓ evaluation/{file}")

    print(f"\n🚀 Next steps:")
    print(f"   1. Test the Streamlit app locally:")
    print(f"      streamlit run app.py")
    print(f"   2. Deploy to Streamlit Cloud:")
    print(f"      Follow instructions in DEPLOYMENT.md")

    return best_model, results


if __name__ == "__main__":
    best_model, results = main()
