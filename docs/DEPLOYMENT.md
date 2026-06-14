# Streamlit Deployment Guide

This guide covers deploying your Product Price Prediction app to Streamlit Cloud.

## Prerequisites

- GitHub account
- Streamlit account (sign up at https://share.streamlit.io)
- Your project pushed to a GitHub repository
- Model artifacts (`model.pkl` and `preprocessor.pkl` in `./artifacts/`)

## Quick Start - Local Testing

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Locally
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Deployment Options

### Option 1: Streamlit Cloud (Recommended - Free)

#### Step 1: Push Code to GitHub
```bash
git init
git add .
git commit -m "Initial commit: Product price predictor"
git push -u origin main
```

#### Step 2: Deploy to Streamlit Cloud
1. Go to https://share.streamlit.io
2. Sign in with your GitHub account
3. Click **"New app"**
4. Configure:
   - **Repository**: Select your repo
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **"Deploy"**

#### Step 3: Add Secrets (if needed)
In Streamlit Cloud dashboard:
1. Click your app
2. Go to **Settings** → **Secrets**
3. Add any API keys or credentials in TOML format

### Option 2: Docker Deployment

#### Create Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Build and Run
```bash
docker build -t product-price-predictor .
docker run -p 8501:8501 product-price-predictor
```

### Option 3: Heroku Deployment

#### Step 1: Add `Procfile`
```
web: streamlit run app.py --logger.level=error --server.port=$PORT --server.address=0.0.0.0
```

#### Step 2: Deploy
```bash
heroku create your-app-name
git push heroku main
```

### Option 4: AWS Deployment

Use AWS App Runner or EC2 with similar Docker setup.

## Pre-Deployment Checklist

- [ ] Model artifacts in `./artifacts/` directory
- [ ] All dependencies in `requirements.txt`
- [ ] `.gitignore` configured (secrets excluded)
- [ ] `.streamlit/config.toml` created
- [ ] Code pushed to GitHub
- [ ] README.md updated with instructions
- [ ] Test locally: `streamlit run app.py`

## Important Notes

### Artifacts
If model files are large (>100MB), you may need to:
- Use Git LFS for model storage
- Host models separately (S3, GCS) and download on startup
- Use Streamlit caching effectively

### Performance
- Models are cached with `@st.cache_resource`
- First load may be slow; subsequent requests are fast
- Consider using a smaller model for production if needed

### Security
- Never commit `.streamlit/secrets.toml`
- Use environment variables for sensitive data
- Enable HTTPS on deployment

## Troubleshooting

### "Model not found" Error
```python
# Check that artifacts exist:
ls -la ./artifacts/
# Should show: model.pkl, preprocessor.pkl
```

### Large Upload Size
Increase in `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 200  # in MB
```

### Slow Performance
- Optimize model size
- Use Streamlit caching
- Consider async operations for heavy processing

## Monitoring

### View Logs (Streamlit Cloud)
1. Go to app dashboard
2. Click **Logs**
3. Check for errors

### Local Debugging
```bash
streamlit run app.py --logger.level=debug
```

## Support

- Streamlit Docs: https://docs.streamlit.io/
- Streamlit Community: https://discuss.streamlit.io/
- GitHub Issues: https://github.com/streamlit/streamlit/issues

---

**Next Steps:**
1. Ensure model artifacts are ready
2. Push code to GitHub
3. Deploy via Streamlit Cloud
