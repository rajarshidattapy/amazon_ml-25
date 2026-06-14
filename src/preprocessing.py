"""Data preprocessing module for product data."""

import pandas as pd
import numpy as np
import re
import warnings
import gc
from typing import Tuple, Dict, List, Any
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings('ignore')


class RobustProductDataPreprocessor:
    """Robust preprocessor for product catalog data with text and categorical features."""

    def __init__(self, max_text_length: int = 1000, max_tfidf_features: int = 1000):
        """
        Initialize the preprocessor.

        Args:
            max_text_length: Maximum length for text fields
            max_tfidf_features: Maximum number of TF-IDF features to extract
        """
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

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if pd.isna(text):
            return ""

        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^\w\s\.]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def extract_structured_features(self, catalog_content: str) -> Dict[str, Any]:
        """Extract structured components from catalog content."""
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

        # Extract title
        if 'Item Name:' in text:
            try:
                title_part = text.split('Item Name:')[1].split('Bullet Point')[0].split('Product Description')[0].strip()
                features['title'] = self.clean_text(title_part)
            except:
                pass

        # Extract bullet points
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

        # Extract description
        if 'Product Description:' in text:
            try:
                desc_part = text.split('Product Description:')[1].split('Value:')[0].strip()
                features['description'] = self.clean_text(desc_part[:500])
            except:
                pass

        # Extract value and unit
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

        # Extract Item Pack Quantity (IPQ)
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

        # Extract brand from title
        if features['title']:
            words = features['title'].split()
            if len(words) > 1 and len(words[0]) > 2:
                features['brand'] = words[0]

        return features

    def extract_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic text-based features."""
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

    def create_light_tfidf_features(self, df: pd.DataFrame, is_training: bool = True):
        """Create TF-IDF features from combined text."""
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

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with sensible defaults."""
        df['value'] = df['value'].fillna(1.0)
        df['ipq'] = df['ipq'].fillna(1.0)

        text_columns = ['title', 'description', 'brand', 'unit']
        for col in text_columns:
            df[col] = df[col].fillna('')

        return df

    def create_essential_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from base features."""
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

    def robust_encode_categorical_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Encode categorical features with robust handling of unseen categories."""
        if is_training:
            if 'brand' in df.columns:
                df['brand_encoded'] = self.brand_encoder.fit_transform(df['brand'])
                self.fitted_brands = set(df['brand'].unique())

            df['unit_encoded'] = self.unit_encoder.fit_transform(df['unit'])
            self.fitted_units = set(df['unit'].unique())
        else:
            # Handle unseen brands
            if 'brand' in df.columns:
                df['brand'] = df['brand'].apply(lambda x: x if x in self.fitted_brands else 'unknown')
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

        # One-hot encode unit categories
        essential_categories = ['weight', 'volume', 'count']
        for category in essential_categories:
            df[f'unit_cat_{category}'] = (df['unit_category'] == category).astype(int)

        return df

    def scale_numerical_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Scale numerical features using StandardScaler."""
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

    def preprocess_data_robust(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete preprocessing pipeline.

        Args:
            df: Input dataframe
            is_training: Whether this is training data (fits encoders) or validation/test

        Returns:
            Tuple of (feature_dataframe, processed_dataframe)
        """
        # Extract structured features
        structured_features = df['catalog_content'].apply(self.extract_structured_features)

        essential_columns = ['title', 'bullet_points', 'description', 'value', 'unit', 'brand', 'ipq']
        for col in essential_columns:
            df[col] = structured_features.apply(lambda x: x[col])

        df['bullet_points_combined'] = df['bullet_points'].apply(
            lambda x: ' '.join(x) if isinstance(x, list) else ''
        )

        df = self.handle_missing_values(df)
        df = self.extract_basic_features(df)

        # TF-IDF features
        tfidf_matrix = self.create_light_tfidf_features(df, is_training)

        df = self.create_essential_derived_features(df)
        df = self.robust_encode_categorical_features(df, is_training)
        df = self.scale_numerical_features(df, is_training)

        # Compile feature matrix
        feature_columns = [
            'value', 'ipq', 'title_length', 'description_length', 'num_bullet_points',
            'price_per_unit', 'brand_popularity', 'brand_encoded', 'unit_encoded'
        ] + [col for col in df.columns if col.startswith('is_')] + \
          [col for col in df.columns if col.startswith('unit_cat_')]

        feature_columns = [col for col in feature_columns if col in df.columns]

        numerical_features = df[feature_columns].copy()

        # Combine with TF-IDF
        tfidf_columns = [f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_columns, index=df.index)

        final_features = pd.concat([numerical_features, tfidf_df], axis=1)

        del tfidf_matrix, numerical_features, tfidf_df
        gc.collect()

        return final_features, df
