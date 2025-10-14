# Amazon ML Challenge - Product Price Prediction

A comprehensive machine learning solution for predicting product prices using multimodal features (text + images) with robust preprocessing and ensemble modeling.

## 🎯 Project Overview

This project implements an end-to-end pipeline for product price prediction using:
- **Text Features**: Product titles, descriptions, bullet points extracted from catalog content
- **Image Features**: Deep learning features extracted from product images using ResNet50
- **Ensemble Models**: Multiple ML algorithms combined for optimal performance
- **Cross-Validation**: 5-fold validation with mixed feature types

## 📊 Key Results

- **Best Performance**: ~XX% SMAPE (Symmetric Mean Absolute Percentage Error)
- **Feature Engineering**: 800+ text features + 2048 image features
- **Model Types**: LightGBM, XGBoost, Random Forest, Gradient Boosting, Ridge, Lasso
- **Ensemble Methods**: Weighted ensemble based on validation performance

## 🚀 Quick Start

### Prerequisites
```bash
# Required libraries
pip install pandas numpy scikit-learn
pip install lightgbm xgboost
pip install torch torchvision
pip install nltk pillow requests
pip install matplotlib seaborn tqdm
```

### Google Colab Setup
```python
# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Upload your dataset
from google.colab import files
uploaded = files.upload()  # Upload your dataset.zip
```

### Basic Usage
```python
# 1. First time - Process and save data
train_data, test_data, folds = main()

# 2. Future sessions - Load preprocessed data
train_data, test_data, folds = load_from_google_drive()

# 3. Train models
results = train_on_all_folds_with_mixed_features(PREPROCESSED_DIR, n_splits=5)
```

## 📁 Project Structure

```
├── AmazonMLChallenge.ipynb          # Main notebook
├── README.md                        # This file
├── /content/drive/MyDrive/product_price_prediction/
│   ├── kfold_splits/               # Cross-validation splits
│   ├── preprocessed_data_robust/   # Processed features
│   ├── parsed_data/               # Parsed catalog content
│   └── visualizations/            # Data analysis plots
├── saved_models_mixed/            # Trained models
└── advanced_results_mixed_features/ # Performance results
```

## 🔧 Pipeline Components

### 1. Data Preprocessing (`RobustProductDataPreprocessor`)
- **Text Cleaning**: Removes HTML, URLs, special characters
- **Feature Extraction**: Titles, descriptions, bullet points, brands, units
- **TF-IDF Vectorization**: 1000 most important text features
- **Categorical Encoding**: Robust handling of unseen categories
- **Numerical Scaling**: StandardScaler for numerical features

### 2. Image Feature Extraction (`OptimizedImageFeatureExtractor`)
- **Model**: Pre-trained ResNet50 (2048 features per image)
- **GPU Acceleration**: Automatic GPU detection and usage
- **Batch Processing**: Optimized for memory efficiency
- **Error Handling**: Graceful handling of failed image downloads

### 3. Model Training
- **Individual Models**: 6 different algorithms
- **Ensemble Methods**: Weighted and simple average ensembles
- **Cross-Validation**: 5-fold with mixed feature types
- **Performance Metrics**: SMAPE, MAE for comprehensive evaluation

## 📈 Feature Engineering

### Text Features (800+ features)
- **Basic Features**: Title length, description length, bullet point count
- **TF-IDF Features**: 1000 most important terms
- **Derived Features**: Price per unit, brand popularity
- **Category Features**: Product type classification (food, beverage, personal care)

### Image Features (2048 features)
- **CNN Features**: ResNet50 pre-trained on ImageNet
- **Preprocessing**: Resize, center crop, normalization
- **Batch Processing**: Efficient GPU utilization

### Combined Features
- **Text + Image**: Concatenated feature vectors
- **Feature Alignment**: Robust handling of missing image features
- **Scaling**: Standardized numerical features

## 🎯 Model Performance

### Individual Models
| Model | Avg SMAPE | Std SMAPE |
|-------|-----------|-----------|
| LightGBM | XX.XX% | X.XX% |
| XGBoost | XX.XX% | X.XX% |
| Random Forest | XX.XX% | X.XX% |
| Gradient Boosting | XX.XX% | X.XX% |
| Ridge | XX.XX% | X.XX% |
| Lasso | XX.XX% | X.XX% |

### Ensemble Performance
- **Weighted Ensemble**: XX.XX% SMAPE
- **Simple Average**: XX.XX% SMAPE

## 💾 Data Storage Strategy

### Google Drive Integration
- **Persistent Storage**: All processed data saved to Google Drive
- **Session Recovery**: Load preprocessed data in future sessions
- **Automatic Backup**: Models and results automatically saved

### File Organization
```
/content/drive/MyDrive/product_price_prediction/
├── kfold_splits/                   # K-fold cross-validation splits
│   ├── fold_0/ ... fold_4/        # Individual fold data
│   └── fold_info.json             # Fold metadata
├── preprocessed_data_robust/       # Processed features
│   ├── fold_0/ ... fold_4/        # Per-fold processed data
│   └── fitted_preprocessor.pkl    # Trained preprocessor
├── parsed_data/                   # Parsed catalog content
└── visualizations/                # Analysis plots
```

## 🔄 Cross-Validation Strategy

### Mixed Feature Approach
- **Folds 0-1**: Image + Text features (if image extraction completed)
- **Folds 2-4**: Text-only features (fallback for incomplete image processing)
- **Robust Alignment**: Automatic feature alignment between folds

### Validation Metrics
- **Primary**: SMAPE (Symmetric Mean Absolute Percentage Error)
- **Secondary**: MAE (Mean Absolute Error)
- **Cross-Validation**: 5-fold with stratification option

## 🛠️ Advanced Features

### Robust Preprocessing
- **Missing Value Handling**: Intelligent imputation strategies
- **Categorical Encoding**: Handles unseen categories in validation/test
- **Text Processing**: NLTK-based cleaning and lemmatization
- **Feature Selection**: TF-IDF with optimal parameters

### GPU Optimization
- **Automatic Detection**: GPU/CPU automatic selection
- **Memory Management**: Efficient batch processing
- **Error Recovery**: Graceful fallback to CPU if GPU fails

### Ensemble Learning
- **Weighted Ensemble**: Performance-based model weighting
- **Model Diversity**: Multiple algorithm types for robust predictions
- **Cross-Validation**: Ensemble weights optimized per fold

## 📊 Usage Examples

### Complete Pipeline
```python
# 1. Data Processing
train_data, test_data, folds = main()

# 2. Feature Extraction (Text + Images)
preprocessor = preprocess_folds_robust('/content/kfold_splits', n_folds=5)

# 3. Image Feature Extraction (if GPU available)
extract_remaining_folds_optimized()

# 4. Model Training
results = train_on_all_folds_with_mixed_features(PREPROCESSED_DIR, n_splits=5)

# 5. Load Best Models
best_models = load_best_models('./saved_models_mixed/')
```

### Quick Training (Text Only)
```python
# Train with text features only
results = train_on_folds_0_and_1(PREPROCESSED_DIR)
```

### Load Preprocessed Data
```python
# Load from Google Drive in future sessions
train_data, test_data, folds = load_from_google_drive()
preprocessor = load_preprocessor_from_drive()
```

## 🔍 Monitoring and Debugging

### Progress Tracking
- **Fold Completion**: Automatic progress tracking
- **Feature Status**: Mixed feature type monitoring
- **Performance Metrics**: Real-time SMAPE tracking

### Error Handling
- **Image Download Failures**: Zero-feature fallback
- **GPU Memory Issues**: Automatic batch size adjustment
- **Missing Data**: Robust imputation strategies

## 📋 Requirements

### Python Libraries
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
lightgbm>=3.2.0
xgboost>=1.4.0
torch>=1.9.0
torchvision>=0.10.0
nltk>=3.6.0
pillow>=8.3.0
requests>=2.25.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
```

### Hardware Recommendations
- **RAM**: 16GB+ recommended for full pipeline
- **GPU**: CUDA-compatible GPU for image processing (optional)
- **Storage**: 10GB+ for processed data and models

## 🎯 Performance Tips

### Memory Optimization
- Use batch processing for large datasets
- Clear GPU cache between folds
- Save intermediate results to disk

### Speed Optimization
- Use GPU for image feature extraction
- Parallel processing for text features
- Efficient data loading strategies

### Accuracy Optimization
- Combine text and image features when possible
- Use ensemble methods for robust predictions
- Optimize hyperparameters per fold

