# Cell 0
from google.colab import drive
import os

drive.mount('/content/drive')

# Define your permanent storage paths
DRIVE_BASE_PATH = '/content/drive/MyDrive/AmazonML_Challenge/'

# Cell 1
import os
import pandas as pd
import numpy as np
import re
from tqdm.notebook import tqdm

# Cell 2
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Cell 3
from PIL import Image
import requests
from io import BytesIO
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

# Cell 4
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import make_scorer
import lightgbm as lgb

# Cell 5
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Cell 6
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Cell 7
def smape(y_true, y_pred):
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100

smape_scorer = make_scorer(smape, greater_is_better=False)

# Cell 8
ZIP_FILE_PATH = '/content/sample_data/68e8d1d70b66d_student_resource.zip'

# Cell 9
import os
file_path = '/content/sample_data/68e8d1d70b66d_student_resource.zip'
if os.path.exists(file_path) and file_path.endswith('.zip'):
    ZIP_FILE_PATH = file_path
else:
    # Look for CSV files instead
    csv_files = [f for f in os.listdir('/content/sample_data') if f.endswith('.csv')]
    if csv_files:
        ZIP_FILE_PATH = f'/content/sample_data/{csv_files[0]}'
    else:
        raise FileNotFoundError("No dataset files found")

# Cell 10
import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json
from google.colab import drive

def extract_and_analyze_data(zip_path, extract_to='./dataset'):
    """
    Extract data from zip file and perform basic analysis
    """
    print("📦 Extracting dataset...")

    # Create directory if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)

    # Extract zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    print("✅ Extraction completed!")

    # Load training data
    train_path = os.path.join(extract_to, 'train.csv')
    test_path = os.path.join(extract_to, 'test.csv')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.csv not found in {extract_to}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None

    return train_df, test_df, extract_to

def parse_catalog_content(df):
    """
    Parse catalog_content into structured components
    """
    print("🔍 Parsing catalog_content...")

    def extract_components(text):
        if pd.isna(text):
            return {
                'title': '',
                'bullet_points': [],
                'description': '',
                'value': np.nan,
                'unit': ''
            }

        components = {
            'title': '',
            'bullet_points': [],
            'description': '',
            'value': np.nan,
            'unit': ''
        }

        # Extract Item Name
        if 'Item Name:' in text:
            try:
                title_part = text.split('Item Name:')[1].split('Bullet Point')[0].split('Product Description')[0].strip()
                components['title'] = title_part
            except:
                pass

        # Extract Bullet Points
        if 'Bullet Point' in text:
            try:
                bullet_section = text.split('Bullet Point')[1:]
                for bullet in bullet_section[:6]:  # Take first 6 bullet points
                    bullet_text = bullet.split(':', 1)[1].split('Bullet Point')[0].split('Product Description')[0].strip()
                    if bullet_text:
                        components['bullet_points'].append(bullet_text)
            except:
                pass

        # Extract Product Description
        if 'Product Description:' in text:
            try:
                desc_part = text.split('Product Description:')[1].split('Value:')[0].strip()
                components['description'] = desc_part
            except:
                pass

        # Extract Value and Unit
        if 'Value:' in text and 'Unit:' in text:
            try:
                value_part = text.split('Value:')[1].split('Unit:')[0].strip()
                unit_part = text.split('Unit:')[1].strip()

                # Handle multiple lines in unit
                unit_part = unit_part.split('\n')[0].split('"')[0].strip()

                # Convert value to float if possible
                try:
                    components['value'] = float(value_part)
                except:
                    components['value'] = np.nan

                components['unit'] = unit_part
            except:
                pass

        return components

    # Apply parsing
    parsed_data = df['catalog_content'].apply(extract_components)

    # Create new columns
    df['title'] = parsed_data.apply(lambda x: x['title'])
    df['bullet_points'] = parsed_data.apply(lambda x: x['bullet_points'])
    df['description'] = parsed_data.apply(lambda x: x['description'])
    df['value'] = parsed_data.apply(lambda x: x['value'])
    df['unit'] = parsed_data.apply(lambda x: x['unit'])
    df['num_bullet_points'] = parsed_data.apply(lambda x: len(x['bullet_points']))
    df['title_length'] = df['title'].str.len()
    df['description_length'] = df['description'].str.len()

    return df

def analyze_data(train_df, test_df=None):
    """
    Perform comprehensive data analysis
    """
    print("📊 Analyzing dataset...")

    analysis = {}

    # Basic statistics
    analysis['basic_stats'] = {
        'train_samples': len(train_df),
        'test_samples': len(test_df) if test_df is not None else 0,
        'price_min': train_df['price'].min(),
        'price_max': train_df['price'].max(),
        'price_mean': train_df['price'].mean(),
        'price_median': train_df['price'].median(),
        'price_std': train_df['price'].std()
    }

    # Missing values
    analysis['missing_values'] = train_df.isnull().sum().to_dict()

    # Unit distribution
    analysis['unit_distribution'] = train_df['unit'].value_counts().head(10).to_dict()

    # Value statistics
    analysis['value_stats'] = {
        'value_min': train_df['value'].min(),
        'value_max': train_df['value'].max(),
        'value_mean': train_df['value'].mean(),
        'value_median': train_df['value'].median()
    }

    # Text length statistics
    analysis['text_stats'] = {
        'avg_title_length': train_df['title_length'].mean(),
        'avg_description_length': train_df['description_length'].mean(),
        'avg_bullet_points': train_df['num_bullet_points'].mean()
    }

    return analysis

def create_kfold_splits(train_df, n_splits=5, strategy='kfold', random_state=42):
    """
    Create K-fold splits for cross-validation

    Parameters:
    - train_df: Training dataframe
    - n_splits: Number of folds
    - strategy: 'kfold' or 'stratified' (if we can create meaningful strata)
    - random_state: Random seed for reproducibility
    """
    print(f"🔄 Creating {n_splits}-fold splits...")

    # Reset index to ensure consistent indexing
    train_df = train_df.reset_index(drop=True)

    # Create folds
    if strategy == 'stratified':
        # Create price bins for stratification (adjust bins based on your price distribution)
        price_bins = pd.cut(train_df['price'], bins=10, labels=False)
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(kf.split(train_df, price_bins))
    else:
        # Regular KFold
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(kf.split(train_df))

    # Create fold datasets
    fold_datasets = {}

    for fold, (train_idx, val_idx) in enumerate(splits):
        fold_datasets[fold] = {
            'train': train_df.iloc[train_idx].copy(),
            'val': train_df.iloc[val_idx].copy()
        }

        print(f"Fold {fold}: Train samples = {len(train_idx)}, Val samples = {len(val_idx)}")

    return fold_datasets, splits

def save_fold_splits(fold_datasets, save_dir):
    """
    Save fold splits to disk
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save each fold
    for fold, data in fold_datasets.items():
        fold_dir = os.path.join(save_dir, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)

        data['train'].to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
        data['val'].to_csv(os.path.join(fold_dir, 'val.csv'), index=False)

    # Save fold indices for reference
    fold_info = {
        'total_folds': len(fold_datasets),
        'fold_sizes': {fold: {'train': len(data['train']), 'val': len(data['val'])}
                      for fold, data in fold_datasets.items()}
    }

    with open(os.path.join(save_dir, 'fold_info.json'), 'w') as f:
        json.dump(fold_info, f, indent=2)

    print(f"✅ Fold splits saved to {save_dir}")

def save_to_google_drive(fold_datasets, train_df_parsed, test_df_parsed, extract_dir):
    """
    Save all data to Google Drive for permanent storage
    """
    print("💾 Saving to Google Drive...")

    # Define Google Drive paths
    DRIVE_BASE_PATH = '/content/drive/MyDrive/product_price_prediction/'
    DRIVE_FOLDS_DIR = os.path.join(DRIVE_BASE_PATH, 'kfold_splits')
    DRIVE_PARSED_DIR = os.path.join(DRIVE_BASE_PATH, 'parsed_data')

    # Create directories in Drive
    os.makedirs(DRIVE_FOLDS_DIR, exist_ok=True)
    os.makedirs(DRIVE_PARSED_DIR, exist_ok=True)

    # Save fold splits to Drive
    for fold, data in fold_datasets.items():
        fold_dir = os.path.join(DRIVE_FOLDS_DIR, f'fold_{fold}')
        os.makedirs(fold_dir, exist_ok=True)

        data['train'].to_csv(os.path.join(fold_dir, 'train.csv'), index=False)
        data['val'].to_csv(os.path.join(fold_dir, 'val.csv'), index=False)

    # Save parsed datasets to Drive
    train_df_parsed.to_csv(os.path.join(DRIVE_PARSED_DIR, 'train_parsed.csv'), index=False)
    if test_df_parsed is not None:
        test_df_parsed.to_csv(os.path.join(DRIVE_PARSED_DIR, 'test_parsed.csv'), index=False)

    # Save fold info to Drive
    fold_info = {
        'total_folds': len(fold_datasets),
        'fold_sizes': {fold: {'train': len(data['train']), 'val': len(data['val'])}
                      for fold, data in fold_datasets.items()}
    }

    with open(os.path.join(DRIVE_FOLDS_DIR, 'fold_info.json'), 'w') as f:
        json.dump(fold_info, f, indent=2)

    print(f"✅ All data saved to Google Drive:")
    print(f"   - Fold splits: {DRIVE_FOLDS_DIR}")
    print(f"   - Parsed data: {DRIVE_PARSED_DIR}")

def visualize_data(train_df, save_dir):
    """
    Create visualizations for data understanding
    """
    print("📈 Creating visualizations...")

    os.makedirs(save_dir, exist_ok=True)

    # Price distribution
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.hist(train_df['price'], bins=50, alpha=0.7, edgecolor='black')
    plt.title('Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Frequency')

    # Log price distribution (if prices have wide range)
    plt.subplot(2, 2, 2)
    log_prices = np.log1p(train_df['price'])
    plt.hist(log_prices, bins=50, alpha=0.7, edgecolor='black')
    plt.title('Log Price Distribution')
    plt.xlabel('Log(Price + 1)')
    plt.ylabel('Frequency')

    # Value distribution
    plt.subplot(2, 2, 3)
    plt.hist(train_df['value'].dropna(), bins=50, alpha=0.7, edgecolor='black')
    plt.title('Value Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Unit distribution (top 10)
    plt.subplot(2, 2, 4)
    top_units = train_df['unit'].value_counts().head(10)
    top_units.plot(kind='bar', alpha=0.7)
    plt.title('Top 10 Unit Types')
    plt.xlabel('Unit')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'data_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Text length distributions
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.hist(train_df['title_length'], bins=50, alpha=0.7, edgecolor='black')
    plt.title('Title Length Distribution')
    plt.xlabel('Title Length (chars)')

    plt.subplot(1, 3, 2)
    plt.hist(train_df['description_length'], bins=50, alpha=0.7, edgecolor='black')
    plt.title('Description Length Distribution')
    plt.xlabel('Description Length (chars)')

    plt.subplot(1, 3, 3)
    plt.hist(train_df['num_bullet_points'], bins=20, alpha=0.7, edgecolor='black')
    plt.title('Number of Bullet Points')
    plt.xlabel('Bullet Points Count')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'text_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Also save visualizations to Google Drive
    DRIVE_VIS_DIR = '/content/drive/MyDrive/product_price_prediction/visualizations'
    os.makedirs(DRIVE_VIS_DIR, exist_ok=True)

    # Copy visualizations to Drive
    import shutil
    for file in ['data_distributions.png', 'text_distributions.png']:
        src = os.path.join(save_dir, file)
        dst = os.path.join(DRIVE_VIS_DIR, file)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    print(f"✅ Visualizations also saved to Google Drive: {DRIVE_VIS_DIR}")

def main():
    """
    Main execution function
    """
    # Mount Google Drive first
    print("🔗 Mounting Google Drive...")
    drive.mount('/content/drive')
    print("✅ Google Drive mounted!")

    # Configuration
    ZIP_FILE_PATH = '/content/sample_data/68e8d1d70b66d_student_resource.zip'  # Update this path in Colab
    EXTRACT_DIR = '/content/dataset/student_resource/dataset'
    FOLDS_DIR = './kfold_splits'
    N_SPLITS = 5
    RANDOM_STATE = 42

    try:
        # Step 1: Extract data
        train_df, test_df, extract_dir = extract_and_analyze_data(ZIP_FILE_PATH, EXTRACT_DIR)

        # Step 2: Parse catalog content
        train_df_parsed = parse_catalog_content(train_df)
        if test_df is not None:
            test_df_parsed = parse_catalog_content(test_df)
        else:
            test_df_parsed = None

        # Step 3: Analyze data
        analysis = analyze_data(train_df_parsed, test_df)

        print("\n" + "="*50)
        print("📊 DATA ANALYSIS SUMMARY")
        print("="*50)
        for section, stats in analysis.items():
            print(f"\n{section.upper().replace('_', ' ')}:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

        # Step 4: Create K-fold splits
        fold_datasets, splits = create_kfold_splits(
            train_df_parsed,
            n_splits=N_SPLITS,
            random_state=RANDOM_STATE
        )

        # Step 5: Save folds locally
        save_fold_splits(fold_datasets, FOLDS_DIR)

        # Step 6: Save everything to Google Drive
        save_to_google_drive(fold_datasets, train_df_parsed, test_df_parsed, extract_dir)

        # Step 7: Create visualizations
        visualize_data(train_df_parsed, './visualizations')

        # Step 8: Save parsed datasets locally
        train_df_parsed.to_csv(os.path.join(extract_dir, 'train_parsed.csv'), index=False)
        if test_df is not None:
            test_df_parsed.to_csv(os.path.join(extract_dir, 'test_parsed.csv'), index=False)

        print("\n🎉 PROCESS COMPLETED SUCCESSFULLY!")
        print(f"📁 Original data: {EXTRACT_DIR}")
        print(f"📁 Local K-fold splits: {FOLDS_DIR}")
        print(f"📁 Local visualizations: ./visualizations")
        print(f"📁 PERMANENT Google Drive storage: /content/drive/MyDrive/product_price_prediction/")
        print(f"📁 Parsed datasets saved with '_parsed.csv' suffix")

        return train_df_parsed, test_df_parsed, fold_datasets

    except Exception as e:
        print(f"❌ Error: {e}")
        raise

# Function to load data from Google Drive in future sessions
def load_from_google_drive():
    """
    Load preprocessed data from Google Drive in future Colab sessions
    """
    print("📥 Loading data from Google Drive...")

    # Mount Google Drive
    drive.mount('/content/drive')

    DRIVE_BASE_PATH = '/content/drive/MyDrive/product_price_prediction/'
    DRIVE_FOLDS_DIR = os.path.join(DRIVE_BASE_PATH, 'kfold_splits')
    DRIVE_PARSED_DIR = os.path.join(DRIVE_BASE_PATH, 'parsed_data')

    # Check if data exists in Drive
    if not os.path.exists(DRIVE_FOLDS_DIR):
        print("❌ No preprocessed data found in Google Drive. Run main() first.")
        return None, None, None

    # Load fold datasets
    fold_datasets = {}
    n_folds = len([f for f in os.listdir(DRIVE_FOLDS_DIR) if f.startswith('fold_')])

    for fold_num in range(n_folds):
        fold_dir = os.path.join(DRIVE_FOLDS_DIR, f'fold_{fold_num}')
        if os.path.exists(fold_dir):
            train_df = pd.read_csv(os.path.join(fold_dir, 'train.csv'))
            val_df = pd.read_csv(os.path.join(fold_dir, 'val.csv'))
            fold_datasets[fold_num] = {'train': train_df, 'val': val_df}

    # Load parsed datasets
    train_parsed_path = os.path.join(DRIVE_PARSED_DIR, 'train_parsed.csv')
    test_parsed_path = os.path.join(DRIVE_PARSED_DIR, 'test_parsed.csv')

    train_df_parsed = pd.read_csv(train_parsed_path) if os.path.exists(train_parsed_path) else None
    test_df_parsed = pd.read_csv(test_parsed_path) if os.path.exists(test_parsed_path) else None

    print("✅ Data loaded successfully from Google Drive!")
    return train_df_parsed, test_df_parsed, fold_datasets

# Usage instructions:
def print_usage_instructions():
    print("\n" + "="*60)
    print("📋 USAGE INSTRUCTIONS")
    print("="*60)
    print("FIRST TIME:")
    print("1. Run: train_data, test_data, folds = main()")
    print("   This will process data and save to Google Drive")
    print("\nFUTURE SESSIONS:")
    print("2. Run: train_data, test_data, folds = load_from_google_drive()")
    print("   This will load preprocessed data from Google Drive")
    print("="*60)

# Usage in Colab:
if __name__ == "__main__":
    # First, upload your dataset.zip to Colab, then run:
    # from google.colab import files
    # uploaded = files.upload()  # Upload your dataset.zip

    # Print usage instructions
    print_usage_instructions()

    # Then run the main processing
    train_data, test_data, folds = main()

# Cell 11
dataset_path = '/content/dataset/student_resource/dataset'
train_df = pd.read_csv(os.path.join(dataset_path, '/content/kfold_splits/fold_0/train.csv'))
test_df = pd.read_csv(os.path.join(dataset_path, '/content/kfold_splits/fold_0/val.csv'))
sample_submission_df = pd.read_csv(os.path.join(dataset_path, 'sample_test_out.csv'))
print(f'Train data shape: {train_df.shape}')
print(f'Test data shape: {test_df.shape}')
print(f'Sample submission shape: {sample_submission_df.shape}')

print('\nTrain Data Head:')
print(train_df.head())
print('\nTest Data Head:')
print(test_df.head())

# Cell 12
import nltk
nltk.download('punkt_tab')

# Cell 13
import os
import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import warnings
import gc
import joblib
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

warnings.filterwarnings('ignore')

# Download NLTK resources
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
except:
    pass

# Define Google Drive paths
DRIVE_BASE_PATH = '/content/drive/MyDrive/product_price_prediction/'
PREPROCESSED_DIR = os.path.join(DRIVE_BASE_PATH, 'preprocessed_data_robust')

class RobustProductDataPreprocessor:
    def __init__(self, max_text_length=1000, max_tfidf_features=1000):
        self.max_text_length = max_text_length
        self.max_tfidf_features = max_tfidf_features
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        self.brand_encoder = LabelEncoder()
        self.unit_encoder = LabelEncoder()
        self.fitted_brands = set()
        self.fitted_units = set()

    def clean_text(self, text):
        if pd.isna(text):
            return ""

        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^\w\s\.]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def extract_structured_features(self, catalog_content):
        features = {
            'title': '',
            'bullet_points': [],
            'description': '',
            'value': np.nan,
            'unit': '',
            'brand': '',
            'ipq': np.nan,
        }

        if pd.isna(catalog_content):
            return features

        text = str(catalog_content)

        if 'Item Name:' in text:
            try:
                title_part = text.split('Item Name:')[1].split('Bullet Point')[0].split('Product Description')[0].strip()
                features['title'] = self.clean_text(title_part)
            except:
                pass

        if 'Bullet Point' in text:
            try:
                bullet_section = text.split('Bullet Point')[1:]
                for bullet in bullet_section[:5]:
                    if ':' in bullet:
                        bullet_text = bullet.split(':', 1)[1].split('Bullet Point')[0].split('Product Description')[0].strip()
                        cleaned_bullet = self.clean_text(bullet_text)
                        if cleaned_bullet:
                            features['bullet_points'].append(cleaned_bullet)
            except:
                pass

        if 'Product Description:' in text:
            try:
                desc_part = text.split('Product Description:')[1].split('Value:')[0].strip()
                features['description'] = self.clean_text(desc_part[:500])
            except:
                pass

        if 'Value:' in text and 'Unit:' in text:
            try:
                value_part = text.split('Value:')[1].split('Unit:')[0].strip()
                unit_part = text.split('Unit:')[1].strip()
                unit_part = unit_part.split('\n')[0].split('"')[0].strip().lower()

                try:
                    features['value'] = float(value_part)
                except:
                    features['value'] = np.nan

                features['unit'] = unit_part
            except:
                pass

        ipq_patterns = [
            r'Item Pack Quantity.*?(\d+\.?\d*)',
            r'IPQ.*?(\d+\.?\d*)',
            r'pack.*?of.*?(\d+\.?\d*)',
        ]

        for pattern in ipq_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    features['ipq'] = float(matches[0])
                    break
                except:
                    pass

        if features['title']:
            words = features['title'].split()
            if len(words) > 1 and len(words[0]) > 2:
                features['brand'] = words[0]

        return features

    def extract_basic_features(self, df):
        print("📝 Extracting basic features...")

        df['combined_text'] = (df['title'] + ' ' + df['description']).str[:800]

        df['title_length'] = df['title'].str.len()
        df['description_length'] = df['description'].str.len()
        df['num_bullet_points'] = df['bullet_points'].apply(len)

        product_categories = {
            'food': ['chocolate', 'candy', 'snack', 'tea', 'coffee', 'spice'],
            'beverage': ['water', 'soda', 'juice', 'drink', 'beverage'],
            'personal_care': ['shampoo', 'soap', 'lotion', 'deodorant'],
        }

        for category, keywords in product_categories.items():
            df[f'is_{category}'] = df['combined_text'].apply(
                lambda x: any(keyword in x for keyword in keywords)
            ).astype(int)

        return df

    def create_light_tfidf_features(self, df, is_training=True):
        print("🔤 Creating light TF-IDF features...")

        if is_training:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_tfidf_features,
                stop_words='english',
                ngram_range=(1, 1),
                min_df=5,
                max_df=0.9,
                dtype=np.float32
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(df['combined_text'])
        else:
            tfidf_matrix = self.tfidf_vectorizer.transform(df['combined_text'])

        return tfidf_matrix

    def handle_missing_values(self, df):
        print("🔧 Handling missing values...")

        df['value'] = df['value'].fillna(1.0)
        df['ipq'] = df['ipq'].fillna(1.0)

        text_columns = ['title', 'description', 'brand', 'unit']
        for col in text_columns:
            df[col] = df[col].fillna('')

        return df

    def create_essential_derived_features(self, df):
        print("🎯 Creating essential derived features...")

        df['price_per_unit'] = df['price'] / df['value']
        df['price_per_unit'] = df['price_per_unit'].replace([np.inf, -np.inf], 0)
        df['price_per_unit'] = df['price_per_unit'].fillna(0)

        if 'brand' in df.columns:
            brand_counts = df['brand'].value_counts()
            df['brand_popularity'] = df['brand'].map(brand_counts).fillna(1)

        df['unit_category'] = 'other'
        weight_units = ['ounce', 'oz', 'pound', 'lb']
        volume_units = ['fl oz', 'fluid ounce', 'liter']
        count_units = ['count', 'ct', 'pack']

        for unit in weight_units:
            mask = df['unit'].str.contains(unit, case=False, na=False)
            df.loc[mask, 'unit_category'] = 'weight'

        for unit in volume_units:
            mask = df['unit'].str.contains(unit, case=False, na=False)
            df.loc[mask, 'unit_category'] = 'volume'

        for unit in count_units:
            mask = df['unit'].str.contains(unit, case=False, na=False)
            df.loc[mask, 'unit_category'] = 'count'

        return df

    def robust_encode_categorical_features(self, df, is_training=True):
        print("🔠 Encoding categorical features robustly...")

        if is_training:
            # Fit encoders and store seen categories
            if 'brand' in df.columns:
                df['brand_encoded'] = self.brand_encoder.fit_transform(df['brand'])
                self.fitted_brands = set(df['brand'].unique())

            df['unit_encoded'] = self.unit_encoder.fit_transform(df['unit'])
            self.fitted_units = set(df['unit'].unique())
        else:
            # Handle unseen categories in validation/test data
            if 'brand' in df.columns:
                # Replace unseen brands with 'unknown'
                df['brand'] = df['brand'].apply(lambda x: x if x in self.fitted_brands else 'unknown')
                # If 'unknown' wasn't in training, add it to encoder
                if 'unknown' not in self.fitted_brands and (df['brand'] == 'unknown').any():
                    all_brands = list(self.fitted_brands) + ['unknown']
                    self.brand_encoder = LabelEncoder()
                    self.brand_encoder.fit(all_brands)
                    self.fitted_brands = set(all_brands)
                df['brand_encoded'] = self.brand_encoder.transform(df['brand'])

            # Handle unseen units
            df['unit'] = df['unit'].apply(lambda x: x if x in self.fitted_units else 'unknown')
            if 'unknown' not in self.fitted_units and (df['unit'] == 'unknown').any():
                all_units = list(self.fitted_units) + ['unknown']
                self.unit_encoder = LabelEncoder()
                self.unit_encoder.fit(all_units)
                self.fitted_units = set(all_units)
            df['unit_encoded'] = self.unit_encoder.transform(df['unit'])

        # One-hot encode essential unit categories
        essential_categories = ['weight', 'volume', 'count']
        for category in essential_categories:
            df[f'unit_cat_{category}'] = (df['unit_category'] == category).astype(int)

        return df

    def scale_numerical_features(self, df, is_training=True):
        print("⚖️ Scaling numerical features...")

        numerical_columns = [
            'value', 'ipq', 'title_length', 'description_length',
            'num_bullet_points', 'price_per_unit', 'brand_popularity'
        ]

        numerical_columns = [col for col in numerical_columns if col in df.columns]

        if is_training:
            df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])
        else:
            df[numerical_columns] = self.scaler.transform(df[numerical_columns])

        return df

    def preprocess_data_robust(self, df, is_training=True):
        print("🚀 Starting robust preprocessing...")

        print("1. Extracting structured features...")
        structured_features = df['catalog_content'].apply(self.extract_structured_features)

        essential_columns = ['title', 'bullet_points', 'description', 'value', 'unit', 'brand', 'ipq']
        for col in essential_columns:
            df[col] = structured_features.apply(lambda x: x[col])

        df['bullet_points_combined'] = df['bullet_points'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else ''
        )

        df = self.handle_missing_values(df)
        df = self.extract_basic_features(df)

        print("4. Creating TF-IDF features...")
        tfidf_matrix = self.create_light_tfidf_features(df, is_training)

        df = self.create_essential_derived_features(df)
        df = self.robust_encode_categorical_features(df, is_training)
        df = self.scale_numerical_features(df, is_training)

        feature_columns = [
            'value', 'ipq', 'title_length', 'description_length', 'num_bullet_points',
            'price_per_unit', 'brand_popularity', 'brand_encoded', 'unit_encoded'
        ] + [col for col in df.columns if col.startswith('is_')] + \
          [col for col in df.columns if col.startswith('unit_cat_')]

        feature_columns = [col for col in feature_columns if col in df.columns]

        numerical_features = df[feature_columns].copy()

        tfidf_columns = [f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_columns, index=df.index)

        final_features = pd.concat([numerical_features, tfidf_df], axis=1)

        print(f"✅ Preprocessing completed! Final feature shape: {final_features.shape}")

        del tfidf_matrix, numerical_features, tfidf_df
        gc.collect()

        return final_features, df

def preprocess_folds_robust(folds_dir, n_folds=5):
    print("🔄 Preprocessing folds with robust encoding...")

    # Use Google Drive for permanent storage
    output_dir = PREPROCESSED_DIR
    os.makedirs(output_dir, exist_ok=True)

    print(f"💾 Saving preprocessed data to Google Drive: {output_dir}")

    # Fit preprocessor on first 2 folds to capture most categories
    print("📊 Fitting preprocessor on first 2 folds...")
    all_train_data = []

    for fold_num in range(min(2, n_folds)):  # Use first 2 folds for fitting
        train_path = os.path.join(folds_dir, f'fold_{fold_num}', 'train.csv')
        train_df = pd.read_csv(train_path)
        all_train_data.append(train_df)

    combined_train = pd.concat(all_train_data, ignore_index=True)

    # Use a reasonable sample size
    sample_size = min(50000, len(combined_train))
    sample_df = combined_train.sample(n=sample_size, random_state=42)

    print(f"Fitting sample shape: {sample_df.shape}")

    preprocessor = RobustProductDataPreprocessor(
        max_text_length=800,
        max_tfidf_features=800
    )

    # Fit preprocessor on sample
    sample_features, sample_processed = preprocessor.preprocess_data_robust(sample_df, is_training=True)
    print(f"Preprocessor fitted. Feature shape: {sample_features.shape}")
    print(f"Fitted brands: {len(preprocessor.fitted_brands)}")
    print(f"Fitted units: {len(preprocessor.fitted_units)}")

    # ✅ SAVE PREPROCESSOR TO GOOGLE DRIVE
    preprocessor_path = os.path.join(output_dir, 'fitted_preprocessor.pkl')
    joblib.dump(preprocessor, preprocessor_path)
    print(f"💾 Saved fitted preprocessor to: {preprocessor_path}")

    del all_train_data, combined_train, sample_df, sample_features, sample_processed
    gc.collect()

    # Process each fold
    for fold_num in range(n_folds):
        print(f"\n{'='*50}")
        print(f"Processing Fold {fold_num}")
        print(f"{'='*50}")

        train_path = os.path.join(folds_dir, f'fold_{fold_num}', 'train.csv')
        val_path = os.path.join(folds_dir, f'fold_{fold_num}', 'val.csv')

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

        print(f"Original - Train: {train_df.shape}, Val: {val_df.shape}")

        print("Processing training data...")
        train_features, train_processed = preprocessor.preprocess_data_robust(train_df, is_training=False)

        print("Processing validation data...")
        val_features, val_processed = preprocessor.preprocess_data_robust(val_df, is_training=False)

        print(f"Processed - Train: {train_features.shape}, Val: {val_features.shape}")

        fold_output_dir = os.path.join(output_dir, f'fold_{fold_num}')
        os.makedirs(fold_output_dir, exist_ok=True)

        # Save to Google Drive
        train_features.to_csv(os.path.join(fold_output_dir, 'train_features.csv'), index=False)
        train_processed.to_csv(os.path.join(fold_output_dir, 'train_processed.csv'), index=False)
        val_features.to_csv(os.path.join(fold_output_dir, 'val_features.csv'), index=False)
        val_processed.to_csv(os.path.join(fold_output_dir, 'val_processed.csv'), index=False)

        print(f"✅ Fold {fold_num} saved to Google Drive")

        del train_df, val_df, train_features, train_processed, val_features, val_processed
        gc.collect()

    print(f"\n🎉 All {n_folds} folds processed and saved to Google Drive!")
    print(f"📁 Permanent output directory: {output_dir}")

    return preprocessor

def load_preprocessor_from_drive():
    """Load preprocessor from Google Drive for future sessions"""
    preprocessor_path = os.path.join(PREPROCESSED_DIR, 'fitted_preprocessor.pkl')
    if os.path.exists(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
        print("✅ Preprocessor loaded from Google Drive")
        return preprocessor
    else:
        print("❌ No preprocessor found in Google Drive. Run preprocessing first.")
        return None

def main():
    folds_dir = '/content/kfold_splits'
    n_folds = 5

    print("🔍 Checking fold directories...")
    for fold_num in range(n_folds):
        fold_path = os.path.join(folds_dir, f'fold_{fold_num}')
        if os.path.exists(fold_path):
            train_files = len(os.listdir(fold_path))
            print(f"Fold {fold_num}: ✓ ({train_files} files)")
        else:
            print(f"Fold {fold_num}: ✗ (Missing)")

    preprocessor = preprocess_folds_robust(folds_dir, n_folds)

    return preprocessor

if __name__ == "__main__":
    print("🚀 Starting Text Preprocessing with Google Drive Storage")
    preprocessor = main()

# Cell 14
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
import time

warnings.filterwarnings('ignore')

# Define Google Drive paths
DRIVE_BASE_PATH = '/content/drive/MyDrive/product_price_prediction/'
PREPROCESSED_DIR = os.path.join(DRIVE_BASE_PATH, 'preprocessed_data_robust')

class OptimizedImageFeatureExtractor:
    def __init__(self, model_name='resnet50', feature_dim=2048):
        # Try GPU first, fall back to CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            print("⚠️ Using CPU (GPU not available)")

        self.feature_dim = feature_dim
        self.model = self._load_pretrained_model(model_name)
        self.preprocess = self._get_preprocess_transform()
        self.scaler = StandardScaler()

    def _load_pretrained_model(self, model_name):
        """Load pre-trained CNN model for feature extraction"""
        print(f"Loading {model_name} on {self.device}...")
        model = models.resnet50(pretrained=True)
        # Remove the final classification layer
        model = nn.Sequential(*list(model.children())[:-1])

        model = model.to(self.device)
        model.eval()

        # Use data parallelism if multiple GPUs available
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

        return model

    def _get_preprocess_transform(self):
        """Get image preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def download_and_process_image(self, image_url):
        """Download and process single image"""
        try:
            response = requests.get(image_url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert('RGB')

            # Preprocess
            image_tensor = self.preprocess(image)

            # Extract features
            with torch.no_grad():
                image_tensor = image_tensor.unsqueeze(0).to(self.device)
                features = self.model(image_tensor)
                features = features.squeeze().cpu().numpy().flatten()

            return features

        except Exception as e:
            # Return zero features if image can't be processed
            return np.zeros(self.feature_dim)

    def extract_features_batch_optimized(self, image_urls, batch_size=64):
        """Optimized batch processing with GPU"""
        all_features = []

        # Use larger batch size for GPU
        if self.device.type == 'cuda':
            batch_size = min(batch_size * 2, 128)  # Larger batches for GPU
            print(f"Using batch size {batch_size} for GPU")

        print(f"Extracting features for {len(image_urls)} images on {self.device}...")

        for i in tqdm(range(0, len(image_urls), batch_size)):
            batch_urls = image_urls[i:i+batch_size]
            batch_features = []

            for url in batch_urls:
                features = self.download_and_process_image(url)
                batch_features.append(features)

            all_features.extend(batch_features)

            # Clear GPU cache periodically
            if self.device.type == 'cuda' and i % 1000 == 0:
                torch.cuda.empty_cache()

        return np.array(all_features)

def check_gpu_status():
    """Check GPU availability and memory"""
    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"✅ GPU Available: {torch.cuda.get_device_name()}")
        print(f"   Memory: {gpu_props.total_memory / 1024**3:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        return True
    else:
        print("❌ GPU Not Available - Using CPU")
        return False

def extract_remaining_folds_optimized():
    """Extract image features for remaining folds with GPU optimization"""

    # Check GPU status
    gpu_available = check_gpu_status()

    extractor = OptimizedImageFeatureExtractor()

    # Process only incomplete folds: 2, 3, 4
    remaining_folds = [2, 3, 4]

    print(f"🔄 Processing remaining folds: {remaining_folds}")

    for fold_num in remaining_folds:
        print(f"\n{'='*50}")
        print(f"🔄 Processing Fold {fold_num}")
        print(f"{'='*50}")

        fold_dir = os.path.join(PREPROCESSED_DIR, f'fold_{fold_num}')

        # Check if text features exist
        train_processed_path = os.path.join(fold_dir, 'train_processed.csv')
        train_features_path = os.path.join(fold_dir, 'train_features.csv')

        if not os.path.exists(train_processed_path):
            print(f"❌ Text features not found for fold {fold_num}")
            continue

        # Process training data
        print("📸 Processing training images...")
        train_df = pd.read_csv(train_processed_path)
        print(f"   Found {len(train_df)} training samples")

        # Extract image features with optimized batch size
        batch_size = 64 if gpu_available else 16
        train_image_features = extractor.extract_features_batch_optimized(
            train_df['image_link'].tolist(),
            batch_size=batch_size
        )

        # Save to Google Drive
        train_image_features_path = os.path.join(fold_dir, 'train_image_features.npy')
        np.save(train_image_features_path, train_image_features)
        print(f"💾 Saved training image features: {train_image_features.shape}")

        # Load text features and combine
        train_text_features = pd.read_csv(train_features_path).values
        train_combined = np.hstack([train_text_features, train_image_features])
        train_combined_path = os.path.join(fold_dir, 'train_combined_features.npy')
        np.save(train_combined_path, train_combined)
        print(f"💾 Saved combined training features: {train_combined.shape}")

        # Process validation data
        print("📸 Processing validation images...")
        val_processed_path = os.path.join(fold_dir, 'val_processed.csv')
        val_features_path = os.path.join(fold_dir, 'val_features.csv')

        if not os.path.exists(val_processed_path):
            print(f"❌ Validation data not found for fold {fold_num}")
            continue

        val_df = pd.read_csv(val_processed_path)
        print(f"   Found {len(val_df)} validation samples")

        # Extract image features
        val_image_features = extractor.extract_features_batch_optimized(
            val_df['image_link'].tolist(),
            batch_size=batch_size
        )

        # Save to Google Drive
        val_image_features_path = os.path.join(fold_dir, 'val_image_features.npy')
        np.save(val_image_features_path, val_image_features)
        print(f"💾 Saved validation image features: {val_image_features.shape}")

        # Load text features and combine
        val_text_features = pd.read_csv(val_features_path).values
        val_combined = np.hstack([val_text_features, val_image_features])
        val_combined_path = os.path.join(fold_dir, 'val_combined_features.npy')
        np.save(val_combined_path, val_combined)
        print(f"💾 Saved combined validation features: {val_combined.shape}")

        print(f"✅ Fold {fold_num} completed!")

        # Clear GPU cache between folds
        if gpu_available:
            torch.cuda.empty_cache()

    # Save extractor
    extractor_path = os.path.join(PREPROCESSED_DIR, 'image_extractor.pkl')
    joblib.dump(extractor, extractor_path)
    print(f"💾 Saved image extractor to Google Drive")

    print(f"\n🎉 All remaining folds completed!")

def verify_final_progress():
    """Verify final progress"""
    print(f"\n🔍 Final verification:")
    completed_folds = []
    for fold_num in range(5):
        fold_dir = os.path.join(PREPROCESSED_DIR, f'fold_{fold_num}')
        train_combined = os.path.join(fold_dir, 'train_combined_features.npy')
        val_combined = os.path.join(fold_dir, 'val_combined_features.npy')

        if os.path.exists(train_combined) and os.path.exists(val_combined):
            completed_folds.append(fold_num)
            print(f"   Fold {fold_num}: ✅ COMPLETED")
        else:
            print(f"   Fold {fold_num}: ❌ INCOMPLETE")

    print(f"📊 Final: {len(completed_folds)}/5 folds completed")
    return completed_folds

def estimate_processing_time():
    """Estimate processing time based on device"""
    if torch.cuda.is_available():
        print("⏰ Estimated time with GPU: 1-2 hours per fold")
        print("   Total: 3-6 hours for 3 folds")
    else:
        print("⏰ Estimated time with CPU: 2-3 hours per fold")
        print("   Total: 6-9 hours for 3 folds")

def main_optimized():
    print("🚀 OPTIMIZED IMAGE EXTRACTION")
    print("=" * 50)
    print("📊 Current status: Folds 0-1 ✅ COMPLETED, Folds 2-4 🔄 PROCESSING")
    print("⚡ Using GPU acceleration if available")
    print("=" * 50)

    # Show time estimate
    estimate_processing_time()
    print("=" * 50)

    # Start timer
    start_time = time.time()

    # Extract remaining folds with optimization
    extract_remaining_folds_optimized()

    # Final verification
    completed_folds = verify_final_progress()

    # Calculate time taken
    end_time = time.time()
    hours = (end_time - start_time) / 3600
    print(f"⏱️  Total time: {hours:.2f} hours")

    if len(completed_folds) == 5:
        print("\n🎉 ALL DONE! You can now start training with combined features!")
        print("💡 Use load_combined_features_from_drive() in your training script")
    else:
        print(f"\n⚠️  Still incomplete: {[f for f in range(5) if f not in completed_folds]}")

# Function for training script
def load_combined_features_from_drive(fold_num, data_type='train'):
    """Load combined features from Google Drive for training"""
    fold_dir = os.path.join(PREPROCESSED_DIR, f'fold_{fold_num}')

    if data_type == 'train':
        combined_path = os.path.join(fold_dir, 'train_combined_features.npy')
        processed_path = os.path.join(fold_dir, 'train_processed.csv')
    else:  # validation
        combined_path = os.path.join(fold_dir, 'val_combined_features.npy')
        processed_path = os.path.join(fold_dir, 'val_processed.csv')

    # Load combined features
    if os.path.exists(combined_path):
        X = np.load(combined_path)
        print(f"✓ Loaded COMBINED features from Drive: {X.shape}")
    else:
        raise FileNotFoundError(f"Combined features not found: {combined_path}")

    # Load target
    processed = pd.read_csv(processed_path)
    y = processed['price']

    return X, y

# Run the optimized main function
main_optimized()

# Cell 15
import pandas as pd
import numpy as np

# Define Google Drive paths
DRIVE_BASE_PATH = '/content/drive/MyDrive/product_price_prediction/'
PREPROCESSED_DIR = os.path.join(DRIVE_BASE_PATH, 'preprocessed_data_robust')

print("🔍 Checking if your data exists in Google Drive...")

# Check if the directory exists
if os.path.exists(PREPROCESSED_DIR):
    print(f"✅ Found preprocessed data directory: {PREPROCESSED_DIR}")

    # Check fold progress
    completed_folds = []
    for fold_num in range(5):
        fold_dir = os.path.join(PREPROCESSED_DIR, f'fold_{fold_num}')
        if os.path.exists(fold_dir):
            train_combined = os.path.join(fold_dir, 'train_combined_features.npy')
            if os.path.exists(train_combined):
                completed_folds.append(fold_num)
                print(f"   Fold {fold_num}: ✅ COMPLETED")
            else:
                print(f"   Fold {fold_num}: 🔄 INCOMPLETE")
        else:
            print(f"   Fold {fold_num}: ❌ MISSING")

    print(f"\n📊 Summary: {len(completed_folds)}/5 folds completed")

else:
    print("❌ Preprocessed data not found. You may need to run text preprocessing first.")

# Cell 16
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import joblib

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def load_preprocessed_fold_robust(preprocessed_dir, fold_num, data_type='train'):
    fold_dir = os.path.join(preprocessed_dir, f'fold_{fold_num}')

    if data_type == 'train':
        features_path = os.path.join(fold_dir, 'train_features.csv')
        processed_path = os.path.join(fold_dir, 'train_processed.csv')
    else:  # validation
        features_path = os.path.join(fold_dir, 'val_features.csv')
        processed_path = os.path.join(fold_dir, 'val_processed.csv')

    X = pd.read_csv(features_path)
    processed = pd.read_csv(processed_path)
    y = processed['price']

    return X, y

def train_advanced_models(X_train, y_train, X_val, y_val):
    models = {}
    results = {}

    models['lightgbm'] = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1
    )

    models['xgboost'] = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1
    )

    models['random_forest'] = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    models['gradient_boosting'] = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )

    models['ridge'] = Ridge(alpha=1.0, random_state=42)
    models['lasso'] = Lasso(alpha=0.001, random_state=42, max_iter=2000)

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        train_smape = smape(y_train, y_train_pred)
        val_smape = smape(y_val, y_val_pred)

        train_mae = mean_absolute_error(y_train, y_train_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)

        results[name] = {
            'model': model,
            'train_metrics': {'SMAPE': train_smape, 'MAE': train_mae},
            'val_metrics': {'SMAPE': val_smape, 'MAE': val_mae},
            'predictions': {'train': y_train_pred, 'val': y_val_pred}
        }

        print(f"{name} - Train SMAPE: {train_smape:.2f}%, Val SMAPE: {val_smape:.2f}%")

    return results

def create_ensemble_predictions(results, X_train, X_val, y_train, y_val):
    ensemble_results = {}

    val_predictions = []
    model_weights = []

    for name, result in results.items():
        val_pred = result['predictions']['val']
        val_smape = result['val_metrics']['SMAPE']

        val_predictions.append(val_pred)
        weight = 1.0 / (val_smape + 1e-8)
        model_weights.append(weight)

    val_predictions = np.array(val_predictions)
    model_weights = np.array(model_weights)
    model_weights = model_weights / model_weights.sum()

    weighted_val_pred = np.average(val_predictions, axis=0, weights=model_weights)
    simple_avg_val_pred = np.mean(val_predictions, axis=0)

    weighted_smape = smape(y_val, weighted_val_pred)
    simple_avg_smape = smape(y_val, simple_avg_val_pred)

    ensemble_results['weighted_ensemble'] = {
        'predictions': {'val': weighted_val_pred},
        'val_metrics': {'SMAPE': weighted_smape},
        'model_weights': model_weights
    }

    ensemble_results['simple_ensemble'] = {
        'predictions': {'val': simple_avg_val_pred},
        'val_metrics': {'SMAPE': simple_avg_smape}
    }

    print(f"Weighted Ensemble - Val SMAPE: {weighted_smape:.2f}%")
    print(f"Simple Average Ensemble - Val SMAPE: {simple_avg_smape:.2f}%")

    return ensemble_results

def train_on_folds_0_and_1(preprocessed_dir):
    print("🚀 Training using combined features from Folds 0 and 1...")

    # Load training data from fold 0
    print("Loading Fold 0 training data...")
    X_fold0_train, y_fold0_train = load_preprocessed_fold_robust(preprocessed_dir, 0, 'train')

    # Load training data from fold 1
    print("Loading Fold 1 training data...")
    X_fold1_train, y_fold1_train = load_preprocessed_fold_robust(preprocessed_dir, 1, 'train')

    # Load validation data from fold 1 (using fold 1 as validation set)
    print("Loading Fold 1 validation data...")
    X_val, y_val = load_preprocessed_fold_robust(preprocessed_dir, 1, 'val')

    # Combine training data from folds 0 and 1
    X_train = pd.concat([X_fold0_train, X_fold1_train], ignore_index=True)
    y_train = pd.concat([y_fold0_train, y_fold1_train], ignore_index=True)

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Number of features: {X_train.shape[1]}")

    # Train models
    fold_results = train_advanced_models(X_train, y_train, X_val, y_val)
    ensemble_results = create_ensemble_predictions(fold_results, X_train, X_val, y_train, y_val)

    results = {
        'individual_models': fold_results,
        'ensemble_models': ensemble_results
    }

    best_model_name = min(fold_results.items(), key=lambda x: x[1]['val_metrics']['SMAPE'])[0]
    best_smape = fold_results[best_model_name]['val_metrics']['SMAPE']
    print(f"✅ Training completed. Best model: {best_model_name} ({best_smape:.2f}%)")

    return results

def visualize_results(results, save_dir='./folds_0_1_results'):
    os.makedirs(save_dir, exist_ok=True)

    model_names = list(results['individual_models'].keys())
    val_smapes = [results['individual_models'][name]['val_metrics']['SMAPE'] for name in model_names]

    plt.figure(figsize=(12, 6))

    # Individual model performance
    plt.subplot(1, 2, 1)
    bars = plt.bar(model_names, val_smapes, alpha=0.7, color=['blue', 'orange', 'green', 'red', 'purple', 'brown'])
    plt.title('Individual Models - Validation SMAPE')
    plt.xlabel('Models')
    plt.ylabel('SMAPE (%)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, val_smapes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{value:.2f}%',
                ha='center', va='bottom', fontsize=9)

    # Ensemble performance
    plt.subplot(1, 2, 2)
    ensemble_smapes = [
        results['ensemble_models']['weighted_ensemble']['val_metrics']['SMAPE'],
        results['ensemble_models']['simple_ensemble']['val_metrics']['SMAPE']
    ]
    ensemble_names = ['Weighted\nEnsemble', 'Simple\nAverage']

    bars = plt.bar(ensemble_names, ensemble_smapes, alpha=0.7, color=['green', 'lightgreen'])
    plt.title('Ensemble Models - Validation SMAPE')
    plt.xlabel('Ensemble Methods')
    plt.ylabel('SMAPE (%)')
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, ensemble_smapes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{value:.2f}%',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def print_summary(results):
    print("\n" + "="*70)
    print("🎯 TRAINING SUMMARY (Folds 0 and 1 Combined)")
    print("="*70)

    print("\nIndividual Model Performance:")
    print("-" * 50)

    for name, result in results['individual_models'].items():
        val_smape = result['val_metrics']['SMAPE']
        train_smape = result['train_metrics']['SMAPE']
        print(f"{name:20} | Train SMAPE: {train_smape:6.2f}% | Val SMAPE: {val_smape:6.2f}%")

    print("\nEnsemble Performance:")
    print("-" * 50)

    weighted_smape = results['ensemble_models']['weighted_ensemble']['val_metrics']['SMAPE']
    simple_smape = results['ensemble_models']['simple_ensemble']['val_metrics']['SMAPE']

    print(f"Weighted Ensemble: {weighted_smape:.2f}%")
    print(f"Simple Average Ensemble: {simple_smape:.2f}%")

    best_individual = min(results['individual_models'].items(), key=lambda x: x[1]['val_metrics']['SMAPE'])
    best_ensemble = min(results['ensemble_models'].items(), key=lambda x: x[1]['val_metrics']['SMAPE'])

    print(f"\n🏆 Best Individual Model: {best_individual[0]} ({best_individual[1]['val_metrics']['SMAPE']:.2f}%)")
    print(f"🏆 Best Ensemble Method: {best_ensemble[0]} ({best_ensemble[1]['val_metrics']['SMAPE']:.2f}%)")

def save_models(results, save_dir='./saved_models_folds_0_1'):
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n💾 Saving all trained models...")

    # Save individual models
    for model_name, result in results['individual_models'].items():
        model_path = os.path.join(save_dir, f'{model_name}.pkl')
        joblib.dump(result['model'], model_path)
        print(f"Saved {model_name} to {model_path}")

    # Save ensemble weights
    ensemble_weights = results['ensemble_models']['weighted_ensemble']['model_weights']
    ensemble_info = {
        'model_weights': ensemble_weights,
        'model_names': list(results['individual_models'].keys()),
        'weighted_ensemble_smape': results['ensemble_models']['weighted_ensemble']['val_metrics']['SMAPE'],
        'simple_ensemble_smape': results['ensemble_models']['simple_ensemble']['val_metrics']['SMAPE']
    }
    joblib.dump(ensemble_info, os.path.join(save_dir, 'ensemble_info.pkl'))
    print(f"Saved ensemble info to {os.path.join(save_dir, 'ensemble_info.pkl')}")

def main():
    PREPROCESSED_DIR = '/content/drive/MyDrive/product_price_prediction/preprocessed_data_robust'

    print("🔍 Checking robust preprocessed data for folds 0 and 1...")
    for fold_num in [0, 1]:
        fold_dir = os.path.join(PREPROCESSED_DIR, f'fold_{fold_num}')
        if os.path.exists(fold_dir):
            train_features = os.path.join(fold_dir, 'train_features.csv')
            val_features = os.path.join(fold_dir, 'val_features.csv')
            if os.path.exists(train_features) and os.path.exists(val_features):
                print(f"Fold {fold_num}: ✓ Preprocessed features found")
            else:
                print(f"Fold {fold_num}: ✗ Missing feature files")
        else:
            print(f"Fold {fold_num}: ✗ Fold directory missing")

    # Train using combined folds 0 and 1
    results = train_on_folds_0_and_1(PREPROCESSED_DIR)

    # Visualize results
    visualize_results(results)

    # Print summary
    print_summary(results)

    # Save models
    save_models(results)

    print(f"\n🎉 Training with folds 0 and 1 completed!")
    print(f"📊 Results saved to: ./folds_0_1_results/")
    print(f"💾 Models saved to: ./saved_models_folds_0_1/")

if __name__ == "__main__":
    main()

# Cell 17
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import joblib

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

def load_combined_features(preprocessed_dir, fold_num, data_type='train'):
    fold_dir = os.path.join(preprocessed_dir, f'fold_{fold_num}')

    # Load text features
    if data_type == 'train':
        features_path = os.path.join(fold_dir, 'train_features.csv')
        processed_path = os.path.join(fold_dir, 'train_processed.csv')
        combined_features_path = os.path.join(fold_dir, 'train_combined_features.npy')
        image_features_path = os.path.join(fold_dir, 'train_image_features.npy')
    else:  # validation
        features_path = os.path.join(fold_dir, 'val_features.csv')
        processed_path = os.path.join(fold_dir, 'val_processed.csv')
        combined_features_path = os.path.join(fold_dir, 'val_combined_features.npy')
        image_features_path = os.path.join(fold_dir, 'val_image_features.npy')

    # Load text features
    X_text = pd.read_csv(features_path)
    processed = pd.read_csv(processed_path)
    y = processed['price']

    # Check if image features exist for this fold
    has_image_features = os.path.exists(combined_features_path) and os.path.exists(image_features_path)

    if has_image_features:
        print(f"Fold {fold_num}: Loading image features...")
        # Load combined features (text + image)
        X_combined = np.load(combined_features_path)
        # Load image features separately (if needed)
        X_image = np.load(image_features_path)

        # Convert combined features to DataFrame for consistency
        X_combined_df = pd.DataFrame(X_combined,
                                   columns=[f'combined_feat_{i}' for i in range(X_combined.shape[1])])

        # Reset index to ensure proper concatenation
        X_text = X_text.reset_index(drop=True)
        X_combined_df = X_combined_df.reset_index(drop=True)

        # Combine text features with combined features
        X_final = pd.concat([X_text, X_combined_df], axis=1)
        print(f"Fold {fold_num}: Combined {X_text.shape[1]} text features + {X_combined.shape[1]} combined features")

    else:
        print(f"Fold {fold_num}: Using text features only")
        X_final = X_text

    return X_final, y, has_image_features

def train_advanced_models(X_train, y_train, X_val, y_val):
    models = {}
    results = {}

    models['lightgbm'] = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1
    )

    models['xgboost'] = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1
    )

    models['random_forest'] = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    models['gradient_boosting'] = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )

    models['ridge'] = Ridge(alpha=1.0, random_state=42)
    models['lasso'] = Lasso(alpha=0.001, random_state=42, max_iter=2000)

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        train_smape = smape(y_train, y_train_pred)
        val_smape = smape(y_val, y_val_pred)

        train_mae = mean_absolute_error(y_train, y_train_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)

        results[name] = {
            'model': model,
            'train_metrics': {'SMAPE': train_smape, 'MAE': train_mae},
            'val_metrics': {'SMAPE': val_smape, 'MAE': val_mae},
            'predictions': {'train': y_train_pred, 'val': y_val_pred}
        }

        print(f"{name} - Train SMAPE: {train_smape:.2f}%, Val SMAPE: {val_smape:.2f}%")

    return results

def create_ensemble_predictions(results, X_train, X_val, y_train, y_val):
    ensemble_results = {}

    val_predictions = []
    model_weights = []

    for name, result in results.items():
        val_pred = result['predictions']['val']
        val_smape = result['val_metrics']['SMAPE']

        val_predictions.append(val_pred)
        weight = 1.0 / (val_smape + 1e-8)
        model_weights.append(weight)

    val_predictions = np.array(val_predictions)
    model_weights = np.array(model_weights)
    model_weights = model_weights / model_weights.sum()

    weighted_val_pred = np.average(val_predictions, axis=0, weights=model_weights)
    simple_avg_val_pred = np.mean(val_predictions, axis=0)

    weighted_smape = smape(y_val, weighted_val_pred)
    simple_avg_smape = smape(y_val, simple_avg_val_pred)

    ensemble_results['weighted_ensemble'] = {
        'predictions': {'val': weighted_val_pred},
        'val_metrics': {'SMAPE': weighted_smape},
        'model_weights': model_weights
    }

    ensemble_results['simple_ensemble'] = {
        'predictions': {'val': simple_avg_val_pred},
        'val_metrics': {'SMAPE': simple_avg_smape}
    }

    print(f"Weighted Ensemble - Val SMAPE: {weighted_smape:.2f}%")
    print(f"Simple Average Ensemble - Val SMAPE: {simple_avg_smape:.2f}%")

    return ensemble_results

def train_on_all_folds_with_mixed_features(preprocessed_dir, n_splits=5):
    print(f"🚀 Training on all folds with mixed features (image+text for folds 0-1, text-only for folds 2-4)...")

    all_fold_results = {}

    for current_fold in range(n_splits):
        print(f"\n{'='*50}")
        print(f"Training Fold {current_fold + 1}/{n_splits}")
        print(f"{'='*50}")

        train_folds_features = []
        train_folds_targets = []

        for fold_num in range(n_splits):
            if fold_num != current_fold:
                X_fold, y_fold, has_images = load_combined_features(preprocessed_dir, fold_num, 'train')
                train_folds_features.append(X_fold)
                train_folds_targets.append(y_fold)
                feature_type = "image+text" if has_images else "text-only"
                print(f"  Added Fold {fold_num} ({feature_type}): {X_fold.shape[1]} features")

        X_train = pd.concat(train_folds_features, ignore_index=True)
        y_train = pd.concat(train_folds_targets, ignore_index=True)

        X_val, y_val, val_has_images = load_combined_features(preprocessed_dir, current_fold, 'val')
        val_feature_type = "image+text" if val_has_images else "text-only"

        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Number of features: {X_train.shape[1]}")
        print(f"Validation fold features: {val_feature_type}")

        # Align features between train and validation
        X_train, X_val = align_features(X_train, X_val)

        fold_results = train_advanced_models(X_train, y_train, X_val, y_val)
        ensemble_results = create_ensemble_predictions(fold_results, X_train, X_val, y_train, y_val)

        all_fold_results[current_fold] = {
            'individual_models': fold_results,
            'ensemble_models': ensemble_results,
            'feature_type': val_feature_type
        }

        best_model_name = min(fold_results.items(), key=lambda x: x[1]['val_metrics']['SMAPE'])[0]
        best_smape = fold_results[best_model_name]['val_metrics']['SMAPE']
        print(f"✅ Fold {current_fold} completed. Best model: {best_model_name} ({best_smape:.2f}%)")

    return all_fold_results

def align_features(X_train, X_val):
    """Align features between training and validation sets"""
    # Get common columns
    common_cols = X_train.columns.intersection(X_val.columns)

    # Select only common columns
    X_train_aligned = X_train[common_cols]
    X_val_aligned = X_val[common_cols]

    print(f"Feature alignment: {X_train.shape[1]} -> {len(common_cols)} common features")

    return X_train_aligned, X_val_aligned

def visualize_advanced_results(all_fold_results, save_dir='./advanced_results_mixed_features'):
    os.makedirs(save_dir, exist_ok=True)

    model_names = list(next(iter(all_fold_results.values()))['individual_models'].keys())
    folds = list(all_fold_results.keys())

    plt.figure(figsize=(15, 10))

    for i, model_name in enumerate(model_names):
        val_smapes = [all_fold_results[fold]['individual_models'][model_name]['val_metrics']['SMAPE']
                     for fold in folds]

        plt.subplot(2, 3, i+1)
        bars = plt.bar(range(len(folds)), val_smapes, alpha=0.7)
        plt.title(f'{model_name} - Validation SMAPE')
        plt.xlabel('Fold')
        plt.ylabel('SMAPE (%)')
        plt.xticks(range(len(folds)), [f'Fold {f}' for f in folds])
        plt.grid(True, alpha=0.3)

        # Color bars based on feature type
        for j, fold in enumerate(folds):
            if all_fold_results[fold]['feature_type'] == 'image+text':
                bars[j].set_color('green')
            else:
                bars[j].set_color('blue')

        avg_smape = np.mean(val_smapes)
        plt.axhline(y=avg_smape, color='red', linestyle='--', label=f'Avg: {avg_smape:.2f}%')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Ensemble performance with feature type info
    ensemble_smapes = [all_fold_results[fold]['ensemble_models']['weighted_ensemble']['val_metrics']['SMAPE']
                      for fold in folds]

    plt.figure(figsize=(12, 6))
    colors = ['green' if all_fold_results[fold]['feature_type'] == 'image+text' else 'blue'
              for fold in folds]

    bars = plt.bar(range(len(folds)), ensemble_smapes, alpha=0.7, color=colors)
    plt.title('Weighted Ensemble - Validation SMAPE\n(Green: Image+Text, Blue: Text-only)')
    plt.xlabel('Fold')
    plt.ylabel('SMAPE (%)')
    plt.xticks(range(len(folds)), [f'Fold {f}' for f in folds])
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, ensemble_smapes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{value:.2f}%',
                ha='center', va='bottom')

    avg_ensemble_smape = np.mean(ensemble_smapes)
    plt.axhline(y=avg_ensemble_smape, color='red', linestyle='--', label=f'Avg: {avg_ensemble_smape:.2f}%')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'ensemble_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

def print_advanced_summary(all_fold_results):
    print("\n" + "="*70)
    print("🎯 ADVANCED CROSS-VALIDATION SUMMARY (Mixed Features)")
    print("="*70)

    model_names = list(next(iter(all_fold_results.values()))['individual_models'].keys())

    print("\nModel Performance Summary:")
    print("-" * 50)

    for model_name in model_names:
        val_smapes = [all_fold_results[fold]['individual_models'][model_name]['val_metrics']['SMAPE']
                     for fold in all_fold_results.keys()]

        avg_smape = np.mean(val_smapes)
        std_smape = np.std(val_smapes)

        print(f"{model_name:20} | Avg SMAPE: {avg_smape:6.2f}% | Std: {std_smape:5.2f}%")

    # Separate performance by feature type
    print("\nPerformance by Feature Type:")
    print("-" * 50)

    image_text_folds = [fold for fold in all_fold_results.keys()
                       if all_fold_results[fold]['feature_type'] == 'image+text']
    text_only_folds = [fold for fold in all_fold_results.keys()
                      if all_fold_results[fold]['feature_type'] == 'text-only']

    if image_text_folds:
        image_text_smapes = [all_fold_results[fold]['ensemble_models']['weighted_ensemble']['val_metrics']['SMAPE']
                           for fold in image_text_folds]
        avg_image_text = np.mean(image_text_smapes)
        print(f"Image+Text folds ({len(image_text_folds)}): {avg_image_text:.2f}%")

    if text_only_folds:
        text_only_smapes = [all_fold_results[fold]['ensemble_models']['weighted_ensemble']['val_metrics']['SMAPE']
                          for fold in text_only_folds]
        avg_text_only = np.mean(text_only_smapes)
        print(f"Text-only folds ({len(text_only_folds)}): {avg_text_only:.2f}%")

    ensemble_smapes = [all_fold_results[fold]['ensemble_models']['weighted_ensemble']['val_metrics']['SMAPE']
                      for fold in all_fold_results.keys()]

    avg_ensemble = np.mean(ensemble_smapes)
    std_ensemble = np.std(ensemble_smapes)

    print(f"\n{'Weighted Ensemble':20} | Avg SMAPE: {avg_ensemble:6.2f}% | Std: {std_ensemble:5.2f}%")

    best_fold = np.argmin(ensemble_smapes)
    worst_fold = np.argmax(ensemble_smapes)

    print(f"\nBest Fold: {best_fold} ({all_fold_results[best_fold]['feature_type']}) with SMAPE: {ensemble_smapes[best_fold]:.2f}%")
    print(f"Worst Fold: {worst_fold} ({all_fold_results[worst_fold]['feature_type']}) with SMAPE: {ensemble_smapes[worst_fold]:.2f}%")

def save_best_models(all_fold_results, save_dir='./saved_models_mixed'):
    os.makedirs(save_dir, exist_ok=True)

    # Find the best fold based on ensemble performance
    ensemble_smapes = [all_fold_results[fold]['ensemble_models']['weighted_ensemble']['val_metrics']['SMAPE']
                      for fold in all_fold_results.keys()]
    best_fold = np.argmin(ensemble_smapes)

    print(f"\n💾 Saving models from best fold (Fold {best_fold} - {all_fold_results[best_fold]['feature_type']})...")

    fold_results = all_fold_results[best_fold]['individual_models']

    for model_name, result in fold_results.items():
        model_path = os.path.join(save_dir, f'{model_name}_fold_{best_fold}.pkl')
        joblib.dump(result['model'], model_path)
        print(f"Saved {model_name} to {model_path}")

    # Save ensemble weights
    ensemble_weights = all_fold_results[best_fold]['ensemble_models']['weighted_ensemble']['model_weights']
    ensemble_info = {
        'model_weights': ensemble_weights,
        'best_fold': best_fold,
        'feature_type': all_fold_results[best_fold]['feature_type'],
        'ensemble_smape': ensemble_smapes[best_fold]
    }
    joblib.dump(ensemble_info, os.path.join(save_dir, 'ensemble_info.pkl'))
    print(f"Saved ensemble info to {os.path.join(save_dir, 'ensemble_info.pkl')}")

def main():
    PREPROCESSED_DIR = '/content/preprocessed_data_robust'
    N_SPLITS = 5

    print("🔍 Checking robust preprocessed data with mixed features...")
    for fold_num in range(N_SPLITS):
        fold_dir = os.path.join(PREPROCESSED_DIR, f'fold_{fold_num}')
        if os.path.exists(fold_dir):
            # Check for image features
            train_combined = os.path.join(fold_dir, 'train_combined_features.npy')
            val_combined = os.path.join(fold_dir, 'val_combined_features.npy')
            has_images = os.path.exists(train_combined) and os.path.exists(val_combined)

            feature_type = "image+text" if has_images else "text-only"
            print(f"Fold {fold_num}: ✓ {feature_type} features")
        else:
            print(f"Fold {fold_num}: ✗ Fold directory missing")

    all_results = train_on_all_folds_with_mixed_features(PREPROCESSED_DIR, N_SPLITS)

    visualize_advanced_results(all_results)

    print_advanced_summary(all_results)

    save_best_models(all_results)

    print(f"\n🎉 Advanced training with mixed features completed!")
    print(f"📊 Results saved to: ./advanced_results_mixed_features/")
    print(f"💾 Models saved to: ./saved_models_mixed/")

if __name__ == "__main__":
    main()
