# ms-cs-smart-Integration-ML-system
This project demonstrates end-to-end MLOps practices for deploying a machine learning model that predicts optimal irrigation amounts based on environmental conditions


# 🌾 Smart Irrigation ML System - MLOps Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-2.9+-green.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-brightgreen.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Production-grade MLOps pipeline for predicting irrigation requirements using weather, soil moisture, and crop data. Built with experiment tracking, model versioning, automated deployment, and monitoring.

## 🎯 Project Overview

This project demonstrates end-to-end MLOps practices for deploying a machine learning model that predicts optimal irrigation amounts based on environmental conditions. The system includes:

- **ML Pipeline**: Automated training with hyperparameter tuning
- **Experiment Tracking**: MLflow for model versioning and comparison
- **Model Serving**: FastAPI REST API with Pydantic validation
- **Monitoring**: Real-time prediction tracking and data drift detection
- **CI/CD**: Docker containerization and automated testing
- **Feature Store**: Engineered features with versioning

## 🏗️ Architecture

```
┌─────────────────┐
│  Raw IoT Data   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Feature Engineering    │
│  - Weather features     │
│  - Soil moisture trends │
│  - Crop growth stage    │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│   ML Training Pipeline  │
│   - Model selection     │
│   - Hyperparameter tune │
│   - Cross-validation    │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│   MLflow Tracking       │
│   - Experiment logs     │
│   - Model registry      │
│   - Artifact storage    │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Model Deployment      │
│   - FastAPI service     │
│   - Docker container    │
│   - Health checks       │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│   Monitoring Dashboard  │
│   - Prediction metrics  │
│   - Data drift alerts   │
│   - Performance tracking│
└─────────────────────────┘
```

## 📊 Dataset

Using synthetic irrigation dataset with:
- **10,000+ samples** of historical irrigation data
- **Features**: Temperature, humidity, rainfall, soil moisture, crop type, growth stage
- **Target**: Optimal irrigation amount (liters/hectare)

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.9
docker >= 20.10
```

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/smart-irrigation-mlops.git
cd smart-irrigation-mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Complete Pipeline
```bash
# 1. Generate synthetic data
python src/data/generate_data.py

# 2. Train models with MLflow tracking
python src/training/train_pipeline.py

# 3. Start MLflow UI
mlflow ui --port 5000

# 4. Deploy best model via FastAPI
python src/api/app.py

# 5. Test API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"temperature": 28, "humidity": 65, "soil_moisture": 35, 
       "rainfall": 0, "crop_type": "wheat", "growth_stage": "vegetative"}'
```

### Docker Deployment
```bash
# Build image
docker build -t irrigation-ml-api .

# Run container
docker run -p 8000:8000 irrigation-ml-api

# API available at http://localhost:8000/docs
```

## 📁 Project Structure

```
smart-irrigation-mlops/
├── data/
│   ├── raw/                    # Raw sensor data
│   ├── processed/              # Cleaned and engineered features
│   └── features/               # Feature store snapshots
├── models/
│   ├── trained/                # Serialized models
│   └── registry/               # MLflow model artifacts
├── src/
│   ├── data/
│   │   ├── generate_data.py    # Synthetic data generation
│   │   ├── preprocessing.py    # Data cleaning
│   │   └── feature_eng.py      # Feature engineering
│   ├── training/
│   │   ├── train_pipeline.py   # Training orchestration
│   │   ├── models.py           # Model definitions
│   │   └── evaluate.py         # Model evaluation
│   ├── api/
│   │   ├── app.py              # FastAPI application
│   │   ├── schemas.py          # Pydantic models
│   │   └── predict.py          # Inference logic
│   ├── monitoring/
│   │   ├── drift_detection.py  # Data drift monitoring
│   │   └── metrics.py          # Performance tracking
│   └── utils/
│       ├── config.py           # Configuration management
│       └── logger.py           # Logging setup
├── tests/
│   ├── test_features.py        # Feature engineering tests
│   ├── test_model.py           # Model training tests
│   └── test_api.py             # API endpoint tests
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_feature_selection.ipynb
│   └── 03_model_comparison.ipynb
├── mlruns/                     # MLflow experiment logs
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Multi-service orchestration
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation
└── README.md                   # This file
```

## 🧪 Key Features Demonstrated

### 1. Experiment Tracking with MLflow
```python
with mlflow.start_run(run_name="random_forest_v1"):
    mlflow.log_params(model.get_params())
    mlflow.log_metrics({"rmse": rmse, "r2": r2_score})
    mlflow.sklearn.log_model(model, "model")
```

### 2. Feature Engineering Pipeline
- Rolling averages for soil moisture (7-day, 14-day windows)
- Temperature-humidity interaction features
- Crop growth stage encoding
- Seasonal indicators

### 3. Model Comparison
Trained and compared:
- Linear Regression (baseline)
- Random Forest
- XGBoost
- LightGBM

### 4. FastAPI Serving
```python
@app.post("/predict")
async def predict(request: IrrigationRequest):
    features = prepare_features(request)
    prediction = model.predict(features)
    return {"irrigation_amount": prediction, "unit": "liters/hectare"}
```

### 5. Monitoring Dashboard
- Real-time prediction logging
- Data drift detection using KS-test
- Performance degradation alerts
- Feature importance tracking

## 📈 Results

| Model | RMSE | R² Score | Training Time |
|-------|------|----------|---------------|
| Linear Regression | 45.2 | 0.72 | 0.3s |
| Random Forest | 32.1 | 0.84 | 8.2s |
| **XGBoost** | **28.7** | **0.88** | 12.5s |
| LightGBM | 30.3 | 0.86 | 6.1s |

**Production Model**: XGBoost selected based on R² score and inference latency (<50ms p95)

## 🔍 Model Monitoring

### Data Drift Detection
```bash
python src/monitoring/drift_detection.py --threshold 0.05
```

Monitors:
- Feature distribution changes (KS-test)
- Prediction distribution shifts
- Input data quality (missing values, outliers)

### Performance Tracking
- Daily RMSE on production predictions
- Alert if RMSE degrades >10% from baseline
- Automatic retraining trigger on drift detection

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Test coverage
pytest --cov=src tests/

# Test API
pytest tests/test_api.py -v
```

## 🚢 CI/CD Pipeline

GitHub Actions workflow:
1. **Lint**: Flake8 code quality checks
2. **Test**: Pytest suite with coverage
3. **Build**: Docker image creation
4. **Deploy**: Push to container registry

## 📊 MLflow Experiments

View tracked experiments:
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Access at: `http://localhost:5000`

Compare:
- Hyperparameters across runs
- Evaluation metrics
- Training artifacts
- Model performance curves

## 🔧 Configuration

Edit `src/utils/config.py`:
```python
MODEL_CONFIG = {
    "n_estimators": 100,
    "max_depth": 10,
    "learning_rate": 0.1
}

API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000
}
```

## 📝 API Documentation

Auto-generated docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Example Request
```json
{
  "temperature": 28.5,
  "humidity": 65.0,
  "soil_moisture": 35.2,
  "rainfall": 0.0,
  "crop_type": "wheat",
  "growth_stage": "vegetative",
  "days_since_last_irrigation": 3
}
```

### Example Response
```json
{
  "irrigation_amount": 250.8,
  "unit": "liters/hectare",
  "confidence_interval": [240.2, 261.4],
  "model_version": "v1.2.3",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

## 🎓 MLOps Practices Demonstrated

✅ **Experiment Tracking**: MLflow for reproducibility  
✅ **Model Versioning**: Registry with staging/production tags  
✅ **Automated Testing**: Unit tests, integration tests  
✅ **CI/CD**: Docker, GitHub Actions  
✅ **Monitoring**: Data drift, performance degradation  
✅ **API Design**: RESTful with OpenAPI docs  
✅ **Feature Store**: Versioned feature engineering  
✅ **Containerization**: Docker for deployment  
✅ **Documentation**: Comprehensive README, docstrings  

## 🚀 Future Enhancements

- [ ] Kubernetes deployment with Helm charts
- [ ] A/B testing framework for model comparison
- [ ] Real-time streaming inference (Kafka)
- [ ] Advanced monitoring with Prometheus + Grafana
- [ ] Feature store migration to Feast
- [ ] Model explainability with SHAP
- [ ] Auto-retraining pipeline with Airflow

## 📚 Technologies Used

- **ML**: Scikit-learn, XGBoost, LightGBM
- **MLOps**: MLflow, DVC
- **API**: FastAPI, Pydantic, Uvicorn
- **Monitoring**: Evidently AI, Custom metrics
- **Testing**: Pytest, Coverage.py
- **Containerization**: Docker, Docker Compose
- **CI/CD**: GitHub Actions

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 👤 Author

**Harshit Sachdev**
- LinkedIn: [linkedin.com/in/harshit-sachdev-91482b180](https://linkedin.com/in/harshit-sachdev-91482b180)
- Email: sachdevh1999@gmail.com

## 🙏 Acknowledgments

- Inspired by production MLOps practices at AgAI/Pretlist India
- Dataset generation based on real-world irrigation parameters
- Architecture follows best practices from Google's MLOps whitepaper

---

⭐ **Star this repo if you find it helpful!**