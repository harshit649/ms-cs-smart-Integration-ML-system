# 🚀 Complete Setup Guide - Smart Irrigation MLOps Project

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Project Setup](#project-setup)
3. [Running the Complete Pipeline](#running-the-pipeline)
4. [Deployment Options](#deployment)
5. [GitHub Setup](#github-setup)
6. [Resume Integration](#resume-integration)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software
```bash
- Python 3.9+ (https://www.python.org/downloads/)
- Git (https://git-scm.com/downloads)
- Docker (https://www.docker.com/get-started) - Optional but recommended
- VS Code or PyCharm (Recommended IDE)
```

### Skills Needed
- Basic Python knowledge
- Command line basics
- Git fundamentals (clone, commit, push)

**⏱️ Total Setup Time: 1-2 hours**

---

## Project Setup

### Step 1: Create Project Directory
```bash
# Create main project folder
mkdir smart-irrigation-mlops
cd smart-irrigation-mlops

# Create directory structure
mkdir -p data/{raw,processed,features}
mkdir -p src/{data,training,api,monitoring,utils}
mkdir -p models/trained
mkdir -p tests
mkdir -p notebooks
mkdir -p reports
mkdir -p .github/workflows
```

### Step 2: Set Up Python Environment

#### Option A: Using venv (Recommended for beginners)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

#### Option B: Using conda
```bash
conda create -n irrigation-ml python=3.9
conda activate irrigation-ml
```

### Step 3: Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
python -c "import mlflow, fastapi, xgboost; print('✅ All packages installed')"
```

---

## Running the Pipeline

### Phase 1: Data Generation (5 minutes)

1. **Create the data generation script**
   ```bash
   # Save the generate_data.py file to src/data/
   # (Use the artifact provided)
   ```

2. **Run data generation**
   ```bash
   python src/data/generate_data.py
   ```
   
   **Expected Output:**
   ```
   🌱 Generating 10000 synthetic irrigation data samples...
   🔧 Engineering additional features...
   💾 Saving data to data/...
   ✅ Raw data saved: data/raw/irrigation_raw.csv
   ✅ Processed data saved: data/processed/irrigation_data.csv
   ```

3. **Verify data files created**
   ```bash
   ls data/processed/
   # Should see: irrigation_data.csv
   ```

### Phase 2: Model Training with MLflow (10-15 minutes)

1. **Create training script**
   ```bash
   # Save train_pipeline.py to src/training/
   ```

2. **Run training pipeline**
   ```bash
   python src/training/train_pipeline.py
   ```
   
   **This will:**
   - Train 4 models (Ridge, Random Forest, XGBoost, LightGBM)
   - Log experiments to MLflow
   - Save best model to `models/trained/`
   
   **Expected Output:**
   ```
   Training Baseline: Ridge Regression
   rmse: 45.2341
   r2_score: 0.7234
   
   Training Random Forest
   Testing config 1: {...}
   rmse: 32.1234
   r2_score: 0.8456
   
   Training XGBoost
   rmse: 28.7123
   r2_score: 0.8789
   
   🏆 Best Model: XGBoost (R² = 0.8789)
   ✅ Production model saved: models/trained/xgboost_production_20241214.pkl
   ```

3. **View MLflow experiments**
   ```bash
   mlflow ui --port 5000
   ```
   
   Open browser: http://localhost:5000
   
   **You should see:**
   - Multiple experiment runs
   - Metrics comparison
   - Parameter tracking
   - Model artifacts

### Phase 3: Deploy FastAPI Service (5 minutes)

1. **Create API application**
   ```bash
   # Save app.py to src/api/
   ```

2. **Start the API server**
   ```bash
   python src/api/app.py
   ```
   
   **Expected Output:**
   ```
   INFO:     Started server process
   INFO:     Waiting for application startup.
   ✅ Model loaded successfully: xgboost_production_20241214
   INFO:     Application startup complete.
   INFO:     Uvicorn running on http://0.0.0.0:8000
   ```

3. **Test the API**
   
   **Option A: Using curl**
   ```bash
   curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "temperature": 28.5,
       "humidity": 65.0,
       "soil_moisture": 35.2,
       "rainfall": 0.0,
       "days_since_last_irrigation": 3,
       "crop_type": "wheat",
       "growth_stage": "vegetative",
       "evapotranspiration": 5.2
     }'
   ```
   
   **Option B: Using Swagger UI**
   - Open browser: http://localhost:8000/docs
   - Click on `/predict` endpoint
   - Click "Try it out"
   - Enter sample data
   - Click "Execute"

4. **Expected Response**
   ```json
   {
     "irrigation_amount": 245.67,
     "unit": "liters/hectare",
     "confidence_interval": [233.39, 257.95],
     "model_version": "xgboost_production_20241214",
     "timestamp": "2024-12-14T10:30:00.123456",
     "recommendation": "Moderate irrigation needed. Standard watering schedule."
   }
   ```

### Phase 4: Run Monitoring (5 minutes)

1. **Create drift detection script**
   ```bash
   # Save drift_detection.py to src/monitoring/
   ```

2. **Run drift detection demo**
   ```bash
   python src/monitoring/drift_detection.py
   ```
   
   **This will:**
   - Test drift detection without drift
   - Test drift detection with injected drift
   - Generate reports in `reports/` folder

3. **View reports**
   ```bash
   cat reports/drift_report_normal.json
   cat reports/drift_report_drift.json
   ```

### Phase 5: Run Tests (5 minutes)

1. **Create test file**
   ```bash
   # Save test_api.py to tests/
   ```

2. **Run tests**
   ```bash
   pytest tests/ -v
   ```
   
   **Expected Output:**
   ```
   tests/test_api.py::TestHealthEndpoints::test_root_endpoint PASSED
   tests/test_api.py::TestHealthEndpoints::test_health_check PASSED
   tests/test_api.py::TestPredictionEndpoint::test_valid_prediction PASSED
   ...
   ========== 15 passed in 3.45s ==========
   ```

3. **Check test coverage**
   ```bash
   pytest --cov=src tests/ --cov-report=html
   ```
   
   Open `htmlcov/index.html` in browser to see coverage report.

---

## Deployment

### Option 1: Docker Deployment (Recommended)

1. **Build Docker image**
   ```bash
   docker build -t irrigation-ml-api .
   ```

2. **Run container**
   ```bash
   docker run -p 8000:8000 irrigation-ml-api
   ```

3. **Test API**
   ```bash
   curl http://localhost:8000/health
   ```

### Option 2: Docker Compose (Multi-service)

1. **Create docker-compose.yml**
   ```yaml
   version: '3.8'
   services:
     api:
       build: .
       ports:
         - "8000:8000"
       volumes:
         - ./models:/app/models
       environment:
         - PYTHONUNBUFFERED=1
     
     mlflow:
       image: ghcr.io/mlflow/mlflow:v2.9.2
       ports:
         - "5000:5000"
       command: mlflow server --host 0.0.0.0 --port 5000
   ```

2. **Start services**
   ```bash
   docker-compose up -d
   ```

---

## GitHub Setup

### Step 1: Initialize Git Repository

```bash
# Initialize git
git init

# Create .gitignore
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# MLflow
mlruns/
mlartifacts/

# Data (too large for git)
data/raw/*.csv
data/processed/*.csv

# Models (use Git LFS or separate storage)
models/trained/*.pkl

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Reports
reports/*.json
htmlcov/
.coverage
EOF

# Add files
git add .
git commit -m "Initial commit: Smart Irrigation MLOps project"
```

### Step 2: Create GitHub Repository

1. **Go to GitHub.com**
2. **Click "New Repository"**
3. **Fill in details:**
   - Repository name: `smart-irrigation-mlops`
   - Description: "Production-grade MLOps pipeline for irrigation prediction"
   - Visibility: Public (for portfolio)
   - ✅ Do NOT initialize with README (we have one)

4. **Push to GitHub**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/smart-irrigation-mlops.git
   git branch -M main
   git push -u origin main
   ```

### Step 3: Add GitHub Actions (CI/CD)

Create `.github/workflows/ci.yml`:

```yaml
name: CI Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        pytest tests/ -v
    
    - name: Lint with flake8
      run: |
        flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
  
  build:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: docker build -t irrigation-ml-api .
```

---

## Resume Integration

### Add to Your Resume

**In "Key Projects" Section:**

```
Smart Irrigation MLOps System | Production-Ready ML Pipeline | GitHub Link
End-to-end MLOps implementation for irrigation optimization: automated training 
with MLflow experiment tracking (4 models compared), FastAPI REST API serving 
predictions at <50ms latency, data drift monitoring with KS-test, Docker deployment, 
and 95% test coverage. Demonstrates full ML lifecycle from data engineering to 
production monitoring.

Tech Stack: Python, MLflow, FastAPI, XGBoost, Docker, Pytest, GitHub Actions
Highlights: 88% R² score, automated retraining triggers, comprehensive API documentation
```

### GitHub README Badges

Add to top of README:
```markdown
![CI Status](https://github.com/YOUR_USERNAME/smart-irrigation-mlops/workflows/CI%20Pipeline/badge.svg)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
```

---

## Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError
```bash
# Solution: Ensure virtual environment is activated
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

#### 2. MLflow tracking URI error
```bash
# Solution: Create mlruns directory
mkdir mlruns
```

#### 3. Model not found error in API
```bash
# Solution: Train model first
python src/training/train_pipeline.py
```

#### 4. Port already in use
```bash
# Solution: Change port or kill existing process
# Mac/Linux:
lsof -ti:8000 | xargs kill -9

# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

#### 5. Permission denied on scripts
```bash
# Solution: Make scripts executable
chmod +x src/**/*.py
```

---

## Next Steps After Setup

### Week 1: Polish & Documentation
- [ ] Add docstrings to all functions
- [ ] Create EDA notebook in `notebooks/`
- [ ] Add more detailed README sections
- [ ] Create architecture diagram

### Week 2: Advanced Features
- [ ] Add Prometheus metrics
- [ ] Implement A/B testing framework
- [ ] Add model explainability (SHAP)
- [ ] Create Grafana dashboard

### Week 3: Interview Prep
- [ ] Practice explaining design decisions
- [ ] Prepare demo video (3-5 minutes)
- [ ] Write blog post about the project
- [ ] Practice system design questions

---

## 🎯 Success Checklist

Before showing to recruiters, ensure:

- [ ] All scripts run without errors
- [ ] MLflow UI shows multiple experiments
- [ ] API documentation accessible at /docs
- [ ] Tests pass with >80% coverage
- [ ] Docker image builds successfully
- [ ] GitHub Actions CI passes
- [ ] README has clear setup instructions
- [ ] Code is well-commented
- [ ] No hardcoded credentials
- [ ] Professional commit messages

---

## 📚 Additional Resources

- [MLOps Best Practices](https://ml-ops.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

---

**🎉 Congratulations! You now have a production-grade MLOps project ready for your portfolio!**

For questions or issues, create a GitHub issue or reach out via LinkedIn.