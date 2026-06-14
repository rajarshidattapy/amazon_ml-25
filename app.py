"""Streamlit application for product price prediction."""

import streamlit as st
import os
import joblib
from pathlib import Path
from src.inference import ProductPricePredictor

# Page configuration
st.set_page_config(
    page_title="Product Price Predictor",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model_and_preprocessor():
    """Load model and preprocessor once and cache them."""
    model_path = "./artifacts/model.pkl"
    preprocessor_path = "./artifacts/preprocessor.pkl"

    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        return None, None, "Model or preprocessor not found in ./artifacts/"

    try:
        predictor = ProductPricePredictor(model_path, preprocessor_path)
        return predictor, True, None
    except Exception as e:
        return None, False, str(e)


def main():
    """Main application."""
    st.title("💰 Product Price Prediction System")
    st.markdown(
        "Predict product prices using machine learning based on product catalog content."
    )

    # Load model
    predictor, success, error = load_model_and_preprocessor()

    if not success:
        st.error(f"❌ Failed to load model: {error}")
        st.info("📝 Please ensure the model is trained and saved in ./artifacts/")
        return

    if predictor is None:
        st.warning("⚠️ Model not found. Please train the model first.")
        st.info("Run `python -m src.train` to train the model.")
        return

    # UI Layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 Product Information")
        catalog_content = st.text_area(
            "Paste the product catalog content here:",
            height=200,
            placeholder="Item Name: ...\nBullet Point 1: ...\nProduct Description: ...",
            help="Include Item Name, Bullet Points, Product Description, Value, and Unit"
        )

    with col2:
        st.subheader("ℹ️ Format Guide")
        st.markdown("""
            **Required fields:**
            - Item Name
            - Bullet Points
            - Product Description
            - Value
            - Unit

            **Example:**
            ```
            Item Name: Premium Coffee
            Bullet Point 1: 100% arabica
            Product Description: High quality
            Value: 500
            Unit: grams
            ```
        """)

    # Prediction button
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        predict_button = st.button(
            "🔮 Predict Price",
            use_container_width=True,
            type="primary"
        )

    with col_btn2:
        clear_button = st.button(
            "🗑️ Clear",
            use_container_width=True
        )

    if clear_button:
        st.rerun()

    # Make prediction
    if predict_button:
        if not catalog_content.strip():
            st.error("❌ Please enter product information")
        else:
            with st.spinner("🔄 Processing..."):
                result = predictor.predict({
                    'catalog_content': catalog_content
                })

            if result['success']:
                st.success("✅ Prediction successful!")

                # Display results
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        label="💵 Predicted Price",
                        value=f"${result['prediction']:.2f}",
                        delta=None
                    )

                with col2:
                    st.metric(
                        label="🔢 Features Used",
                        value=result['feature_count']
                    )

                with col3:
                    st.metric(
                        label="📊 Model Status",
                        value="Ready"
                    )

                # Additional info
                st.info(
                    f"📈 Prediction based on {result['feature_count']} engineered features "
                    "from the product catalog content."
                )

            else:
                st.error(f"❌ Prediction failed: {result['error']}")

    # Sidebar information
    with st.sidebar:
        st.markdown("---")
        st.subheader("📋 Information")
        st.markdown("""
            ### About This Tool
            This application predicts product prices using machine learning.

            **Features:**
            - Text feature extraction from product descriptions
            - TF-IDF vectorization
            - Brand and unit analysis
            - Multi-model ensemble predictions

            **Model Metrics:**
            - SMAPE (Symmetric Mean Absolute Percentage Error)
            - R² Score
            - Mean Absolute Error (MAE)

            ### How It Works
            1. Enter product catalog content
            2. System extracts and processes text features
            3. Machine learning model predicts the price
            4. Result displayed with confidence

            ### Data Format
            Ensure your catalog content includes:
            - Product name/title
            - Key features (bullet points)
            - Product description
            - Quantity/value information
            - Unit of measurement
        """)

        st.markdown("---")
        st.caption(
            "💡 For best results, provide complete product information "
            "with clear structure."
        )


if __name__ == "__main__":
    main()
