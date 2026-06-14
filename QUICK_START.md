# Quick Start Guide

## Installation (2 minutes)

```bash
# Navigate to project
cd amazon_ml-25

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

## Training a Model (5 minutes)

You need preprocessed data from the original notebook. Assuming you have:
- `preprocessed_data/fold_0/train_features.csv`
- `preprocessed_data/fold_0/train_processed.csv`
- `preprocessed_data/fold_0/val_features.csv`
- `preprocessed_data/fold_0/val_processed.csv`
- `preprocessed_data/fitted_preprocessor.pkl`

```bash
# Copy the example training script
python train_example.py
```

Or manually:

```python
import numpy as np
import pandas as pd
from src.train import ModelTrainer

# Load data
X_train = pd.read_csv('preprocessed_data/fold_0/train_features.csv').values
y_train = pd.read_csv('preprocessed_data/fold_0/train_processed.csv')['price'].values
X_val = pd.read_csv('preprocessed_data/fold_0/val_features.csv').values
y_val = pd.read_csv('preprocessed_data/fold_0/val_processed.csv')['price'].values

# Train
trainer = ModelTrainer(output_dir='artifacts')
results = trainer.train_all_models(X_train, y_train, X_val, y_val)

# Save best model
trainer.save_best_model()
```

## Run Streamlit App (1 minute)

```bash
streamlit run app.py
```

Then visit: http://localhost:8501

Enter product catalog content and click "Predict Price"

## Make Predictions Programmatically

```python
from src.inference import ProductPricePredictor

# Load model
predictor = ProductPricePredictor(
    'artifacts/model.pkl',
    'artifacts/preprocessor.pkl'
)

# Single prediction
result = predictor.predict({
    'catalog_content': 'Item Name: Coffee Beans\nBullet Point 1: ...\n...'
})

if result['success']:
    print(f"Price: ${result['prediction']:.2f}")
else:
    print(f"Error: {result['error']}")

# Batch predictions
results = predictor.predict_batch([
    {'catalog_content': 'Item 1...'},
    {'catalog_content': 'Item 2...'},
])
```

## Evaluate Model

```python
from src.evaluate import ModelEvaluator
import joblib

evaluator = ModelEvaluator(output_dir='artifacts/evaluation')

# Load model and data
model = joblib.load('artifacts/model.pkl')
X_test = ...  # your test features
y_test = ...  # your test targets

# Full evaluation with visualizations
evaluator.evaluate_and_visualize(model, X_test, y_test, 'my_model')

# Check results
print("Metrics saved to: artifacts/evaluation/evaluation_metrics.json")
print("Report saved to: artifacts/evaluation/evaluation_report.txt")
print("Visualizations in: artifacts/evaluation/")
```

## Preprocess New Data

If you have raw data that needs preprocessing:

```python
import pandas as pd
from src.preprocessing import RobustProductDataPreprocessor
import joblib

# Load training data to fit preprocessor
train_df = pd.read_csv('raw_train.csv')

# Create and fit preprocessor
preprocessor = RobustProductDataPreprocessor()
train_features, train_processed = preprocessor.preprocess_data_robust(
    train_df, 
    is_training=True  # Important: fit on training data
)

# Save preprocessor
joblib.dump(preprocessor, 'artifacts/preprocessor.pkl')

# Later: use on new data
test_df = pd.read_csv('raw_test.csv')
test_features, test_processed = preprocessor.preprocess_data_robust(
    test_df,
    is_training=False  # Important: don't refit on test data
)
```

## Common Tasks

### Extract evaluation metrics

```bash
# View as human-readable report
cat artifacts/evaluation/evaluation_report.txt

# Parse JSON programmatically
import json
with open('artifacts/evaluation/evaluation_metrics.json') as f:
    metrics = json.load(f)
    print(metrics['best_model']['smape'])
```

### Compare models

```python
from src.train import ModelTrainer
import pandas as pd

trainer = ModelTrainer()
results = trainer.train_all_models(X_train, y_train, X_val, y_val)

# Display comparison
df_results = pd.DataFrame([
    {'model': name, **result['val_metrics']}
    for name, result in results.items()
])
print(df_results.sort_values('smape'))
```

### Create custom visualization

```python
import matplotlib.pyplot as plt
import numpy as np

model = ...  # your trained model
y_pred = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Model Predictions')
plt.savefig('custom_plot.png', dpi=300)
```

### Check feature importance (tree-based models)

```python
import pandas as pd
import joblib

# For LightGBM, XGBoost, RandomForest, GradientBoosting
model = joblib.load('artifacts/lightgbm_model.pkl')

importance = model.feature_importances_
feature_names = [f'feature_{i}' for i in range(len(importance))]

df_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

print(df_importance.head(20))
```

## Troubleshooting

### "Module not found" errors

Ensure you're running from the project root:
```bash
cd /path/to/amazon_ml-25
python train_example.py
```

### NLTK data missing

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

### Model/preprocessor not found

Ensure artifacts directory exists and contains:
- `artifacts/model.pkl`
- `artifacts/preprocessor.pkl`

### Out of memory errors

Reduce batch size in preprocessing or train on smaller data subset

### Streamlit app won't start

```bash
# Clear Streamlit cache
rm -rf ~/.streamlit/cache

# Try again
streamlit run app.py
```

## File Locations

```
artifacts/
├── model.pkl                          # Best trained model
├── preprocessor.pkl                   # Fitted preprocessor
├── lightgbm_model.pkl                 # LightGBM model
├── xgboost_model.pkl                  # XGBoost model
├── random_forest_model.pkl            # Random Forest model
├── gradient_boosting_model.pkl        # Gradient Boosting model
├── ridge_model.pkl                    # Ridge model
├── lasso_model.pkl                    # Lasso model
└── evaluation/
    ├── evaluation_metrics.json        # Machine-readable metrics
    ├── evaluation_report.txt          # Human-readable report
    ├── metrics_comparison.png         # Model comparison chart
    ├── predictions_vs_actual.png      # Calibration plot
    ├── residuals.png                  # Residuals analysis
    └── error_distribution.png         # Error histogram
```

## Next Steps

1. **Check PRODUCTION_README.md** for detailed documentation
2. **Check ARCHITECTURE.md** for design details
3. **Review train_example.py** for full training pipeline example
4. **Examine src/** for module implementations

---

**Need help?** Check the main README.md or PRODUCTION_README.md for more details.
