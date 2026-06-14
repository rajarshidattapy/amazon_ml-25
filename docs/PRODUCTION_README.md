# Product Price Prediction - Production Setup

A production-ready machine learning pipeline for predicting product prices using text and image features with ensemble modeling.

## 📋 Project Overview

This project implements an end-to-end ML pipeline for Amazon product price prediction including:
- **Robust data preprocessing** with text cleaning and feature engineering
- **Multi-modal feature extraction** (text + image features)
- **Ensemble modeling** with 6 different algorithms
- **Comprehensive evaluation** with visualization and metrics
- **Production inference** pipeline
- **Interactive Streamlit frontend**

## 📁 Project Structure

```
amazon_ml-25/
├── src/
│   ├── __init__.py
│   ├── utils.py              # Utility functions (SMAPE, metrics)
│   ├── preprocessing.py      # Data preprocessing and feature engineering
│   ├── features.py           # Feature extraction (text, image)
│   ├── train.py              # Model training with multiple algorithms
│   ├── evaluate.py           # Evaluation pipeline and visualizations
│   └── inference.py          # Production inference pipeline
│
├── notebooks/
│   └── main.ipynb            # Original Jupyter notebook
│
├── artifacts/
│   ├── model.pkl             # Trained best model
│   ├── preprocessor.pkl      # Fitted preprocessor
│   └── evaluation/           # Evaluation results
│       ├── evaluation_metrics.json
│       ├── evaluation_report.txt
│       ├── metrics_comparison.png
│       ├── predictions_vs_actual.png
│       ├── residuals.png
│       └── error_distribution.png
│
├── app.py                    # Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # Original project documentation
└── PRODUCTION_README.md      # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to project directory
cd amazon_ml-25

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK resources (required for preprocessing)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('punkt')"
```

### 2. Train the Model

The model training uses preprocessed data. If you have the original notebook processed data:

```bash
# Option A: Use the training module
python -m src.train_pipeline \
    --train_features path/to/train_features.csv \
    --train_targets path/to/train_targets.csv \
    --val_features path/to/val_features.csv \
    --val_targets path/to/val_targets.csv
```

Or create a simple training script:

```python
import numpy as np
import pandas as pd
from src.train import ModelTrainer
from src.preprocessing import RobustProductDataPreprocessor
import joblib

# Load preprocessed data
X_train = pd.read_csv('preprocessed_data/train_features.csv').values
y_train = pd.read_csv('preprocessed_data/train_targets.csv').values.flatten()
X_val = pd.read_csv('preprocessed_data/val_features.csv').values
y_val = pd.read_csv('preprocessed_data/val_targets.csv').values.flatten()

# Load preprocessor
preprocessor = joblib.load('artifacts/preprocessor.pkl')

# Train models
trainer = ModelTrainer(output_dir='artifacts')
results = trainer.train_all_models(X_train, y_train, X_val, y_val)

# Save best model
trainer.save_best_model(metric='smape', output_name='model.pkl')
```

### 3. Evaluate the Model

```bash
# Run evaluation
python -c "
from src.evaluate import ModelEvaluator
import numpy as np
import joblib

evaluator = ModelEvaluator(output_dir='artifacts/evaluation')

# Load model and test data
model = joblib.load('artifacts/model.pkl')
X_test = np.load('test_features.npy')
y_test = np.load('test_targets.npy')

# Evaluate and visualize
evaluator.evaluate_and_visualize(model, X_test, y_test, model_name='ensemble')
"
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 📊 Module Documentation

### `src/utils.py`
Utility functions for metrics calculation:
- `smape()` - Symmetric Mean Absolute Percentage Error
- `calculate_regression_metrics()` - Comprehensive regression metrics (MAE, MSE, RMSE, R², SMAPE)

### `src/preprocessing.py`
`RobustProductDataPreprocessor` class for data preprocessing:
- Text cleaning and normalization
- Structured feature extraction from catalog content
- TF-IDF feature extraction
- Categorical encoding with unseen value handling
- Feature scaling and derived feature creation

**Usage:**
```python
from src.preprocessing import RobustProductDataPreprocessor

preprocessor = RobustProductDataPreprocessor(max_tfidf_features=1000)

# Preprocess training data
train_features, train_processed = preprocessor.preprocess_data_robust(
    train_df, is_training=True
)

# Preprocess validation data
val_features, val_processed = preprocessor.preprocess_data_robust(
    val_df, is_training=False
)
```

### `src/train.py`
`ModelTrainer` class for model training and selection:
- Supports 6 algorithms: LightGBM, XGBoost, Random Forest, Gradient Boosting, Ridge, Lasso
- Easy model comparison
- Automatic model persistence

**Usage:**
```python
from src.train import ModelTrainer

trainer = ModelTrainer(output_dir='artifacts')

# Train all models
results = trainer.train_all_models(
    X_train, y_train, 
    X_val, y_val,
    models_to_train=['lightgbm', 'xgboost']
)

# Get best model
best_name, best_model = trainer.get_best_model(metric='smape')

# Save best model
trainer.save_best_model(output_name='model.pkl')
```

### `src/evaluate.py`
`ModelEvaluator` class for comprehensive evaluation:
- Multiple evaluation metrics
- Visualizations: predictions vs actual, residuals, error distribution
- Metrics comparison across models
- Text report generation

**Usage:**
```python
from src.evaluate import ModelEvaluator

evaluator = ModelEvaluator(output_dir='artifacts/evaluation')

# Evaluate model
metrics = evaluator.evaluate_model(model, X_test, y_test, model_name='best_model')

# Generate all visualizations
evaluator.evaluate_and_visualize(model, X_test, y_test, model_name='best_model')

# Save results
evaluator.save_metrics()
evaluator.generate_report()
```

### `src/inference.py`
`ProductPricePredictor` class for production predictions:
- Input validation
- Single and batch predictions
- Error handling

**Usage:**
```python
from src.inference import ProductPricePredictor

predictor = ProductPricePredictor(
    model_path='artifacts/model.pkl',
    preprocessor_path='artifacts/preprocessor.pkl'
)

# Single prediction
result = predictor.predict({
    'catalog_content': 'Item Name: Coffee\nBullet Point 1: ...\n...'
})

if result['success']:
    print(f"Predicted price: ${result['prediction']:.2f}")
else:
    print(f"Error: {result['error']}")

# Batch predictions
results = predictor.predict_batch(data_list)
```

## 📈 Model Performance

The system evaluates models on:
- **SMAPE** (Symmetric Mean Absolute Percentage Error) - Primary metric
- **R²** - Coefficient of determination
- **MAE** - Mean Absolute Error
- **RMSE** - Root Mean Squared Error

Evaluation outputs are saved in `artifacts/evaluation/`:
- `evaluation_metrics.json` - Detailed metrics
- `evaluation_report.txt` - Text report
- `*.png` - Visualizations

## 🔄 Preprocessing Pipeline

### Text Cleaning
1. Convert to lowercase
2. Remove URLs and HTML tags
3. Remove special characters
4. Normalize whitespace

### Feature Extraction
1. Parse catalog content (title, description, bullet points, value, unit)
2. Extract basic features (text lengths, bullet point count)
3. TF-IDF vectorization (1000 most important terms)
4. Derived features (price per unit, brand popularity)
5. Category features (food, beverage, personal care)
6. Unit categories (weight, volume, count)

### Encoding & Scaling
1. Categorical encoding for brand and unit (with unseen value handling)
2. StandardScaler for numerical features

### Final Feature Set
- Basic numerical features: ~15 features
- TF-IDF features: 1000 features
- **Total: ~1015 features**

## 🎯 Model Training

Supports 6 different regression algorithms:
1. **LightGBM** - Gradient boosting with leaf-wise tree growth
2. **XGBoost** - Optimized gradient boosting
3. **Random Forest** - Ensemble of decision trees
4. **Gradient Boosting** - Sequential tree ensemble
5. **Ridge** - L2 regularized linear regression
6. **Lasso** - L1 regularized linear regression

Automatic model comparison and selection based on validation metrics.

## 🎨 Streamlit Application

Interactive web interface with:
- Text input for product catalog content
- Real-time price prediction
- Feature count display
- Status indicators
- Format guide and examples
- Sidebar information

### Running the App

```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

## 📝 Data Format

### Input Format (Catalog Content)

```
Item Name: [Product Name]
Bullet Point 1: [Feature 1]
Bullet Point 2: [Feature 2]
Bullet Point 3: [Feature 3]
Product Description: [Detailed description]
Value: [Numeric value]
Unit: [Unit of measurement]
```

### Example

```
Item Name: Premium Arabica Coffee Beans
Bullet Point 1: 100% pure arabica beans
Bullet Point 2: Medium roast for smooth flavor
Bullet Point 3: Fresh ground available
Product Description: High-quality coffee beans from Ethiopian highlands
Value: 500
Unit: grams
```

## 🔒 Production Checklist

- [x] Input validation
- [x] Error handling
- [x] Model caching (Streamlit)
- [x] Preprocessing pipeline serialization
- [x] Model serialization (joblib)
- [x] Batch prediction support
- [x] Comprehensive logging
- [x] Metric tracking
- [x] Visualization generation
- [x] Documentation

## 🛠️ Advanced Usage

### Custom Model Training

```python
from src.train import ModelTrainer

trainer = ModelTrainer()

# Train with custom parameters
results = trainer.train_xgboost(
    X_train, y_train,
    n_estimators=300,
    learning_rate=0.01,
    max_depth=7
)
```

### Batch Processing

```python
from src.inference import ProductPricePredictor

predictor = ProductPricePredictor(
    'artifacts/model.pkl',
    'artifacts/preprocessor.pkl'
)

# Process multiple products
products = [
    {'catalog_content': '...'},
    {'catalog_content': '...'},
    # ...
]

results = predictor.predict_batch(products)

for i, result in enumerate(results):
    if result['success']:
        print(f"Product {i}: ${result['prediction']:.2f}")
```

## 📚 Dependencies

All dependencies are listed in `requirements.txt`:
- **Data Processing**: pandas, numpy
- **ML Models**: scikit-learn, lightgbm, xgboost
- **NLP**: nltk
- **Deep Learning**: torch, torchvision (for image features)
- **Visualization**: matplotlib, seaborn
- **Serialization**: joblib
- **Web Framework**: streamlit

## 🐛 Troubleshooting

### NLTK Data Missing
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

### Model Not Found
```bash
# Ensure model is trained and saved in artifacts/
# Run training script or check artifact paths
```

### GPU/CUDA Issues
The code automatically falls back to CPU if CUDA is not available. No configuration needed.

### Memory Issues
If preprocessing large datasets:
1. Reduce batch size in feature extraction
2. Process folds sequentially instead of parallel
3. Clear GPU cache between operations

## 📖 Additional Resources

- Original notebook: `notebooks/main.ipynb`
- Original documentation: `README.md`
- Evaluation results: `artifacts/evaluation/`

## 👥 Contributing

For improvements or bug fixes:
1. Test changes locally with sample data
2. Ensure evaluation metrics are calculated
3. Update relevant documentation
4. Verify Streamlit app functionality

## 📄 License

Same as original project.

---

**Created**: 2024
**Last Updated**: June 2024
**Status**: Production Ready ✅
