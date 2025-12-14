"""
MLflow-based training pipeline for irrigation prediction model.
Demonstrates experiment tracking, hyperparameter tuning, and model registry.
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
from datetime import datetime
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("irrigation_prediction")


def load_data(filepath="data/processed/irrigation_data.csv"):
    """Load preprocessed data with features"""
    df = pd.read_csv(filepath)
    return df


def prepare_features(df):
    """Prepare features and target for training"""
    feature_cols = [
        'temperature', 'humidity', 'soil_moisture', 'rainfall',
        'days_since_last_irrigation', 'evapotranspiration',
        'temp_humidity_index', 'moisture_deficit',
        'soil_moisture_7d_avg', 'soil_moisture_14d_avg',
        'crop_wheat', 'crop_corn', 'crop_rice', 'crop_cotton',
        'stage_vegetative', 'stage_flowering', 'stage_maturity',
        'season_spring', 'season_summer', 'season_fall', 'season_winter'
    ]
    
    X = df[feature_cols]
    y = df['irrigation_amount']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_model(model, X_test, y_test):
    """Calculate evaluation metrics"""
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2,
        'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    }


def train_baseline_model(X_train, y_train, X_test, y_test):
    """Train baseline Ridge regression model"""
    print("\n" + "="*60)
    print("Training Baseline: Ridge Regression")
    print("="*60)
    
    with mlflow.start_run(run_name="ridge_baseline"):
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)
        
        # Log parameters
        mlflow.log_param("model_type", "ridge")
        mlflow.log_param("alpha", 1.0)
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            print(f"{metric_name}: {metric_value:.4f}")
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        return model, metrics


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest with hyperparameter tuning"""
    print("\n" + "="*60)
    print("Training Random Forest")
    print("="*60)
    
    param_grid = [
        {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5},
        {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 10},
        {'n_estimators': 150, 'max_depth': 12, 'min_samples_split': 8},
    ]
    
    best_model = None
    best_score = float('-inf')
    
    for i, params in enumerate(param_grid, 1):
        with mlflow.start_run(run_name=f"random_forest_v{i}"):
            print(f"\nTesting config {i}: {params}")
            
            model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", "random_forest")
            
            # Evaluate
            metrics = evaluate_model(model, X_test, y_test)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                       scoring='r2', n_jobs=-1)
            cv_mean = cv_scores.mean()
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                print(f"{metric_name}: {metric_value:.4f}")
            
            mlflow.log_metric("cv_r2_mean", cv_mean)
            mlflow.log_metric("cv_r2_std", cv_scores.std())
            print(f"CV R² (mean): {cv_mean:.4f} (+/- {cv_scores.std():.4f})")
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Track best model
            if metrics['r2_score'] > best_score:
                best_score = metrics['r2_score']
                best_model = model
    
    return best_model, metrics


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with hyperparameter tuning"""
    print("\n" + "="*60)
    print("Training XGBoost")
    print("="*60)
    
    param_grid = [
        {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
        {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.05},
        {'n_estimators': 150, 'max_depth': 7, 'learning_rate': 0.08},
    ]
    
    best_model = None
    best_score = float('-inf')
    
    for i, params in enumerate(param_grid, 1):
        with mlflow.start_run(run_name=f"xgboost_v{i}"):
            print(f"\nTesting config {i}: {params}")
            
            model = XGBRegressor(
                **params,
                random_state=42,
                n_jobs=-1,
                objective='reg:squarederror'
            )
            model.fit(X_train, y_train)
            
            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", "xgboost")
            
            # Evaluate
            metrics = evaluate_model(model, X_test, y_test)
            
            # Log metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
                print(f"{metric_name}: {metric_value:.4f}")
            
            # Log feature importance
            feature_importance = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            mlflow.log_text(
                feature_importance.to_string(),
                "feature_importance.txt"
            )
            
            # Log model
            mlflow.xgboost.log_model(model, "model")
            
            # Track best model
            if metrics['r2_score'] > best_score:
                best_score = metrics['r2_score']
                best_model = model
                best_metrics = metrics
    
    return best_model, best_metrics


def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train LightGBM model"""
    print("\n" + "="*60)
    print("Training LightGBM")
    print("="*60)
    
    with mlflow.start_run(run_name="lightgbm_v1"):
        params = {
            'n_estimators': 150,
            'max_depth': 8,
            'learning_rate': 0.08,
            'num_leaves': 31
        }
        
        print(f"Parameters: {params}")
        
        model = LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
        model.fit(X_train, y_train)
        
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", "lightgbm")
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            print(f"{metric_name}: {metric_value:.4f}")
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        return model, metrics


def save_production_model(model, model_name="xgboost_production"):
    """Save best model for production deployment"""
    os.makedirs("models/trained", exist_ok=True)
    
    model_path = f"models/trained/{model_name}_{datetime.now().strftime('%Y%m%d')}.pkl"
    joblib.dump(model, model_path)
    
    print(f"\n✅ Production model saved: {model_path}")
    return model_path


def main():
    """Run complete training pipeline"""
    print("\n" + "="*60)
    print("🚀 STARTING IRRIGATION ML TRAINING PIPELINE")
    print("="*60)
    
    # Load data
    print("\n📊 Loading data...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {df.shape[1] - 1}, Samples: {df.shape[0]}")
    
    # Prepare features
    print("\n🔧 Preparing features...")
    X_train, X_test, y_train, y_test = prepare_features(df)
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Train models
    all_results = {}
    
    # 1. Baseline
    model_ridge, metrics_ridge = train_baseline_model(X_train, y_train, X_test, y_test)
    all_results['Ridge'] = metrics_ridge
    
    # 2. Random Forest
    model_rf, metrics_rf = train_random_forest(X_train, y_train, X_test, y_test)
    all_results['Random Forest'] = metrics_rf
    
    # 3. XGBoost
    model_xgb, metrics_xgb = train_xgboost(X_train, y_train, X_test, y_test)
    all_results['XGBoost'] = metrics_xgb
    
    # 4. LightGBM
    model_lgbm, metrics_lgbm = train_lightgbm(X_train, y_train, X_test, y_test)
    all_results['LightGBM'] = metrics_lgbm
    
    # Compare results
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON")
    print("="*60)
    
    comparison_df = pd.DataFrame(all_results).T
    comparison_df = comparison_df.sort_values('r2_score', ascending=False)
    print("\n" + comparison_df.to_string())
    
    # Select best model
    best_model_name = comparison_df.index[0]
    best_r2 = comparison_df.iloc[0]['r2_score']
    
    print(f"\n🏆 Best Model: {best_model_name} (R² = {best_r2:.4f})")
    
    # Save best model for production
    if best_model_name == 'XGBoost':
        best_model = model_xgb
    elif best_model_name == 'LightGBM':
        best_model = model_lgbm
    elif best_model_name == 'Random Forest':
        best_model = model_rf
    else:
        best_model = model_ridge
    
    model_path = save_production_model(best_model, best_model_name.lower().replace(' ', '_'))
    
    print("\n" + "="*60)
    print("✅ TRAINING PIPELINE COMPLETED")
    print("="*60)
    print(f"\n📈 View experiments: mlflow ui --port 5000")
    print(f"🚀 Production model ready: {model_path}")
    print("\n")


if __name__ == "__main__":
    main()