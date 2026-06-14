# Architecture & Design Guide

## Overview

This document explains the architecture of the refactored product price prediction system, design decisions, and how components interact.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                            │
│                       (app.py)                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │ ProductPricePredictor          │
        │ (inference.py)                 │
        │                                │
        │ • Input validation            │
        │ • Preprocessing               │
        │ • Prediction                  │
        └────┬───────────────────────┬───┘
             │                       │
             ▼                       ▼
   ┌──────────────────────┐  ┌──────────────────┐
   │ RobustProduct        │  │ Trained Model    │
   │ DataPreprocessor     │  │                  │
   │ (preprocessing.py)   │  │ • LightGBM       │
   │                      │  │ • XGBoost        │
   │ • Text cleaning      │  │ • RF/GB/etc      │
   │ • Feature extraction │  │ (artifacts/)     │
   │ • Encoding/Scaling   │  │                  │
   └──────────────────────┘  └──────────────────┘
```

## Component Design

### 1. **utils.py** - Utility Functions

**Purpose**: Common utilities used across the pipeline

**Key Functions**:
- `smape()` - Compute Symmetric Mean Absolute Percentage Error (regression metric)
- `calculate_regression_metrics()` - Comprehensive metrics calculation (MAE, MSE, RMSE, R², SMAPE)

**Design Rationale**:
- Centralized metric calculation prevents duplication
- Consistent metric computation across training and evaluation
- Easy to extend with new metrics

### 2. **preprocessing.py** - Data Preprocessing

**Purpose**: Transform raw product data into usable features

**Key Class**: `RobustProductDataPreprocessor`

**Processing Pipeline**:
```
Raw Catalog Content
    │
    ├─ Structured Feature Extraction
    │   ├─ Parse item name, description, bullet points
    │   ├─ Extract value and unit
    │   └─ Extract brand from title
    │
    ├─ Text Cleaning
    │   ├─ Lowercase conversion
    │   ├─ Remove URLs and HTML
    │   └─ Normalize whitespace
    │
    ├─ Basic Feature Extraction
    │   ├─ Text lengths
    │   ├─ Bullet point count
    │   └─ Product categories
    │
    ├─ TF-IDF Vectorization
    │   └─ Extract 1000 most important terms
    │
    ├─ Derived Features
    │   ├─ Price per unit
    │   ├─ Brand popularity
    │   └─ Unit categories (weight/volume/count)
    │
    ├─ Categorical Encoding
    │   ├─ Brand encoding with unseen value handling
    │   └─ Unit encoding with unknown category fallback
    │
    └─ Feature Scaling
        └─ StandardScaler for numerical features
```

**Design Decisions**:

1. **Robust Encoding**: Handles unseen categories in validation/test by mapping to "unknown"
2. **Stateful Preprocessing**: Stores fitted encoders and scaler for inference
3. **Two-phase Processing**: Separates training (`is_training=True`) from inference (`is_training=False`)
4. **Memory Efficiency**: Uses sparse matrices for TF-IDF, cleans up after processing

**Why This Design**:
- Prevents data leakage (validation set doesn't affect encoder fitting)
- Handles production data with previously unseen brands/units
- Separates concerns between training and inference

### 3. **train.py** - Model Training

**Purpose**: Train multiple ML models and compare performance

**Key Class**: `ModelTrainer`

**Supported Models**:
- LightGBM (gradient boosting)
- XGBoost (optimized gradient boosting)
- Random Forest (ensemble of trees)
- Gradient Boosting (sequential tree building)
- Ridge (L2 regularized linear regression)
- Lasso (L1 regularized linear regression)

**Key Methods**:
- `train_all_models()` - Train multiple models in sequence
- `get_best_model()` - Select best model by metric
- `save_best_model()` - Persist best model to disk

**Design Rationale**:
- Multiple models provide robustness through diversity
- Easy model comparison on same dataset
- Automatic model persistence prevents loss of trained models

**Why Each Model**:
- **Tree-based models** (LightGBM, XGBoost, RF, GB): Handle non-linear relationships well
- **Linear models** (Ridge, Lasso): Provide baseline, detect linear patterns
- **Ensemble diversity**: Different algorithms may capture different aspects of data

### 4. **evaluate.py** - Evaluation & Visualization

**Purpose**: Comprehensive model evaluation with metrics and visualizations

**Key Class**: `ModelEvaluator`

**Key Metrics**:
- **SMAPE**: Symmetric Mean Absolute Percentage Error (primary metric)
- **R²**: Coefficient of determination (model fit quality)
- **MAE**: Mean Absolute Error (average prediction error)
- **RMSE**: Root Mean Squared Error (penalizes large errors)

**Visualizations Generated**:
1. **Predictions vs Actual**: Scatter plot showing model calibration
2. **Residuals Analysis**: Shows prediction errors and patterns
3. **Error Distribution**: Histogram of absolute errors
4. **Metrics Comparison**: Bar chart comparing all models

**Output Artifacts**:
- `evaluation_metrics.json` - Machine-readable metrics
- `evaluation_report.txt` - Human-readable report
- `*.png` - All visualizations

**Design Benefits**:
- Multiple metrics catch different aspects of performance
- Visualizations reveal patterns (e.g., heteroscedasticity)
- JSON output enables further analysis and tracking
- Text report suitable for stakeholder communication

### 5. **inference.py** - Production Inference

**Purpose**: Production-ready prediction pipeline

**Key Class**: `ProductPricePredictor`

**Key Methods**:
- `validate_input()` - Validate input structure and content
- `preprocess_input()` - Transform raw input to features
- `predict()` - Single prediction with error handling
- `predict_batch()` - Batch predictions for multiple products

**Design Pattern**:
```
Raw Input
    │
    ├─ Validation (type, structure, content checks)
    │
    ├─ Preprocessing (same pipeline as training)
    │
    ├─ Prediction (model.predict)
    │
    └─ Post-processing (ensure valid output, e.g., non-negative price)
```

**Why This Design**:
- **Input validation**: Catches errors early, provides helpful messages
- **Consistent preprocessing**: Uses same fitted preprocessor as training
- **Graceful error handling**: Returns error messages instead of crashes
- **Batch support**: Efficient processing of multiple items

### 6. **features.py** - Feature Engineering

**Purpose**: Advanced feature extraction utilities

**Key Components**:
- `ImageFeatureExtractor`: Interface for deep learning-based image features
- `combine_text_and_image_features()`: Multi-modal feature alignment

**Design Note**: 
The full image extraction implementation exists in the original notebook's `OptimizedImageFeatureExtractor`. This module provides a clean interface for integration.

### 7. **app.py** - Streamlit Frontend

**Purpose**: Interactive web interface for predictions

**Key Features**:
- `@st.cache_resource`: Model loaded once and reused
- Input validation with user-friendly error messages
- Real-time predictions
- Format guide and examples

**Design Benefits**:
- Cached model prevents reloading on every interaction
- Format guide helps users provide valid input
- Responsive feedback (success/error indicators)
- Sidebar information for self-service help

## Data Flow Diagrams

### Training Pipeline

```
Raw CSV Data
    │
    ▼
[Load Data] (read_csv)
    │
    ├─ Train/Val Split (via folds from original notebook)
    │
    ├─ Training Set                  Validation Set
    │   │                            │
    │   ├─ [Preprocess]              ├─ [Preprocess]
    │   │  (is_training=True)        │  (is_training=False)
    │   │                            │
    │   ├─ [Extract Features]        ├─ [Extract Features]
    │   │                            │
    │   ▼                            ▼
    │   X_train, y_train             X_val, y_val
    │   │                            │
    │   └─────────┬────────────────────────┬─────────┘
    │             │                        │
    │             ├─ [Train All Models]    │
    │             │  ├─ LightGBM          │
    │             │  ├─ XGBoost           │
    │             │  ├─ RandomForest      │
    │             │  ├─ GradientBoosting  │
    │             │  ├─ Ridge             │
    │             │  └─ Lasso             │
    │             │                        │
    │             ├─ [Evaluate Models] ◄──┘
    │             │                        
    │             ├─ [Compare Metrics]    
    │             │                        
    │             ├─ [Select Best Model]  
    │             │                        
    │             ▼                        
    │         artifacts/                   
    │         ├─ model.pkl                
    │         ├─ preprocessor.pkl         
    │         └─ evaluation/               
    │             ├─ metrics.json          
    │             ├─ report.txt            
    │             └─ *.png                 
    │
    └─ [Ready for Production]
```

### Inference Pipeline

```
User Input (Catalog Content)
    │
    ▼
[ProductPricePredictor.predict()]
    │
    ├─ Validate Input
    │   ├─ Check required fields
    │   ├─ Verify non-empty content
    │   └─ Return error if invalid
    │
    ├─ [If valid] Preprocess Input
    │   ├─ Create DataFrame with raw content
    │   ├─ Run through fitted preprocessor
    │   └─ Extract features (same as training)
    │
    ├─ [Model.predict(features)]
    │   └─ Get price prediction
    │
    ├─ Post-process
    │   └─ Ensure non-negative price
    │
    └─ Return Result
        ├─ success: bool
        ├─ prediction: float
        ├─ error: str (if failed)
        └─ feature_count: int
```

## Key Design Decisions

### 1. Separation of Concerns

Each module has a single responsibility:
- **preprocessing.py**: Data transformation
- **train.py**: Model training
- **evaluate.py**: Evaluation metrics and visualization
- **inference.py**: Production predictions
- **utils.py**: Shared utilities

**Benefit**: Easy to test, maintain, and extend individual components

### 2. Stateful Preprocessing

The `RobustProductDataPreprocessor` is stateful (stores fitted encoders, scaler):

**Training Phase**:
```python
preprocessor = RobustProductDataPreprocessor()
features, _ = preprocessor.preprocess_data_robust(df, is_training=True)
# Now preprocessor.scaler, preprocessor.brand_encoder, etc. are fitted
joblib.dump(preprocessor, 'artifacts/preprocessor.pkl')
```

**Inference Phase**:
```python
preprocessor = joblib.load('artifacts/preprocessor.pkl')
features, _ = preprocessor.preprocess_data_robust(new_df, is_training=False)
# Uses fitted scaler and encoders from training
```

**Why**:
- Prevents data leakage (validation data doesn't affect encoder fitting)
- Ensures exact same transformation in production
- Handles unseen categories gracefully

### 3. Model Persistence with joblib

Models and preprocessors are saved using `joblib` (not pickle):

**Why joblib**:
- Handles large NumPy arrays efficiently
- Better compression
- More reliable for scikit-learn objects
- Standard in the ML community

### 4. Type Hints

Most functions include type hints:

```python
def predict(
    self,
    data: Dict[str, Any]
) -> Dict[str, Any]:
```

**Benefits**:
- IDE auto-completion
- Easier to understand expected input/output
- Can catch type errors early
- Improved code documentation

### 5. Error Handling Strategy

**Input Validation**: Early, before expensive operations
**Graceful Degradation**: Return error results instead of crashing
**User-Friendly Messages**: Explain what went wrong and how to fix it

## Extension Points

### Adding a New Model

In `train.py`:
```python
def train_custom_model(self, X_train, y_train, **kwargs):
    model = CustomModel(**kwargs)
    model.fit(X_train, y_train)
    return model

# Add to train_all_models():
trainers['custom_model'] = self.train_custom_model
```

### Adding a New Feature

In `preprocessing.py`:
```python
def create_custom_features(self, df):
    # Extract new feature from data
    df['custom_feature'] = compute_feature(df)
    return df

# Call in preprocess_data_robust():
df = self.create_custom_features(df)
```

### Adding a New Evaluation Metric

In `utils.py`:
```python
def custom_metric(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Use in evaluate.py:
metrics['custom_metric'] = custom_metric(y_test, y_pred)
```

## Performance Considerations

### Memory
- TF-IDF uses sparse matrices (memory efficient)
- Gradient boosting models are lightweight
- Images require ~2048 floats per product (manageable)

### Speed
- Training: ~minutes on modern hardware
- Inference: <100ms per prediction (single product)
- Batch inference: ~10µs per product after model load

### Scalability
- Current design suitable for:
  - Training: 1M+ products (with chunking)
  - Inference: 1000s of predictions per second
- For larger scale:
  - Use distributed training (Ray, Spark)
  - Deploy model with FastAPI/Flask for serving
  - Use model compression for edge deployment

## Testing Strategy

Recommended tests:

1. **Unit Tests** (test individual functions):
   - `test_preprocessing.py`: Verify feature extraction
   - `test_inference.py`: Verify prediction consistency
   - `test_utils.py`: Verify metric calculations

2. **Integration Tests**: 
   - End-to-end: Raw data → prediction

3. **Regression Tests**:
   - Verify metrics on held-out test set
   - Monitor for metric degradation

## Future Enhancements

1. **Hyperparameter Optimization**: Use Optuna/Hyperopt
2. **Feature Selection**: Automatic feature importance-based selection
3. **Model Ensembling**: Weighted ensemble across best models
4. **Online Learning**: Update model incrementally with new data
5. **Explainability**: SHAP values, feature importance visualization
6. **Monitoring**: Track model performance over time
7. **A/B Testing**: Compare new model versions

---

This architecture provides a clean, scalable, and maintainable foundation for the product price prediction system while preserving the quality of the original notebook's ML work.
