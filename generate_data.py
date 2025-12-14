"""
Generate synthetic irrigation dataset for ML training.
Simulates realistic agricultural IoT sensor data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os


def generate_irrigation_data(n_samples=10000, seed=42):
    """
    Generate synthetic irrigation dataset with realistic relationships
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with irrigation data
    """
    np.random.seed(seed)
    
    print(f"🌱 Generating {n_samples} synthetic irrigation data samples...")
    
    # Base features
    data = {}
    
    # Weather conditions
    data['temperature'] = np.random.normal(25, 8, n_samples).clip(5, 45)
    data['humidity'] = np.random.normal(60, 15, n_samples).clip(20, 95)
    data['rainfall'] = np.random.exponential(5, n_samples).clip(0, 100)
    
    # Soil moisture (influenced by rainfall and temperature)
    base_moisture = np.random.normal(45, 15, n_samples)
    rainfall_effect = data['rainfall'] * 0.3
    temp_effect = -(data['temperature'] - 25) * 0.5
    data['soil_moisture'] = (base_moisture + rainfall_effect + temp_effect).clip(10, 90)
    
    # Time-based features
    data['days_since_last_irrigation'] = np.random.poisson(4, n_samples).clip(0, 20)
    
    # Evapotranspiration (influenced by temperature and humidity)
    base_et = 0.0023 * (data['temperature'] + 17.8)
    et_adjustment = (data['temperature'] - data['humidity']/2) ** 0.5
    data['evapotranspiration'] = (base_et * et_adjustment).clip(1, 15)
    
    # Crop types (categorical)
    crop_types = ['wheat', 'corn', 'rice', 'cotton']
    crop_weights = [0.35, 0.25, 0.25, 0.15]  # Wheat more common
    data['crop_type'] = np.random.choice(crop_types, n_samples, p=crop_weights)
    
    # Growth stages (categorical)
    growth_stages = ['vegetative', 'flowering', 'maturity']
    stage_weights = [0.4, 0.35, 0.25]
    data['growth_stage'] = np.random.choice(growth_stages, n_samples, p=stage_weights)
    
    # Season (based on temperature ranges)
    seasons = []
    for temp in data['temperature']:
        if temp < 10:
            seasons.append('winter')
        elif temp < 20:
            seasons.append('spring')
        elif temp < 30:
            seasons.append('summer')
        else:
            seasons.append('summer')  # Hot summer
    data['season'] = seasons
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate target: irrigation_amount (liters/hectare)
    # Complex relationship based on multiple factors
    
    # Base irrigation need
    base_irrigation = 200
    
    # Adjust for soil moisture (inverse relationship)
    moisture_factor = (60 - df['soil_moisture']) * 3  # 60% is optimal
    
    # Adjust for evapotranspiration (direct relationship)
    et_factor = df['evapotranspiration'] * 15
    
    # Adjust for rainfall (inverse relationship)
    rainfall_factor = -df['rainfall'] * 2
    
    # Adjust for days since last irrigation
    days_factor = df['days_since_last_irrigation'] * 8
    
    # Adjust for crop type
    crop_multipliers = {'wheat': 1.0, 'corn': 1.2, 'rice': 1.5, 'cotton': 0.9}
    crop_factor = df['crop_type'].map(crop_multipliers) * 50
    
    # Adjust for growth stage
    stage_multipliers = {'vegetative': 1.0, 'flowering': 1.3, 'maturity': 0.8}
    stage_factor = df['growth_stage'].map(stage_multipliers) * 30
    
    # Combine all factors
    irrigation_need = (
        base_irrigation +
        moisture_factor +
        et_factor +
        rainfall_factor +
        days_factor +
        crop_factor +
        stage_factor
    )
    
    # Add some noise
    noise = np.random.normal(0, 20, n_samples)
    df['irrigation_amount'] = (irrigation_need + noise).clip(50, 800)
    
    return df


def engineer_features(df):
    """Add engineered features to the dataset"""
    
    print("🔧 Engineering additional features...")
    
    # Interaction features
    df['temp_humidity_index'] = df['temperature'] * (100 - df['humidity']) / 100
    df['moisture_deficit'] = (60 - df['soil_moisture']).clip(0, None)
    
    # Rolling averages (simulated)
    df['soil_moisture_7d_avg'] = df['soil_moisture'] * np.random.uniform(0.92, 1.08, len(df))
    df['soil_moisture_14d_avg'] = df['soil_moisture'] * np.random.uniform(0.88, 1.05, len(df))
    
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=['crop_type', 'growth_stage', 'season'], 
                        prefix=['crop', 'stage', 'season'])
    
    return df


def split_and_save(df, output_dir="data"):
    """Split data and save to files"""
    
    print(f"💾 Saving data to {output_dir}/...")
    
    # Create directories
    os.makedirs(f"{output_dir}/raw", exist_ok=True)
    os.makedirs(f"{output_dir}/processed", exist_ok=True)
    
    # Save raw data
    raw_path = f"{output_dir}/raw/irrigation_raw.csv"
    df_raw = df[[
        'temperature', 'humidity', 'rainfall', 'soil_moisture',
        'days_since_last_irrigation', 'evapotranspiration',
        'crop_wheat', 'crop_corn', 'crop_rice', 'crop_cotton',
        'stage_vegetative', 'stage_flowering', 'stage_maturity',
        'irrigation_amount'
    ]]
    df_raw.to_csv(raw_path, index=False)
    print(f"✅ Raw data saved: {raw_path}")
    
    # Save processed data (with all features)
    processed_path = f"{output_dir}/processed/irrigation_data.csv"
    df.to_csv(processed_path, index=False)
    print(f"✅ Processed data saved: {processed_path}")
    
    # Print statistics
    print("\n📊 Dataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Features: {len(df.columns) - 1}")
    print(f"\nIrrigation Amount Statistics:")
    print(df['irrigation_amount'].describe())
    
    return raw_path, processed_path


def visualize_data(df):
    """Print data visualization information"""
    
    print("\n" + "="*60)
    print("📈 DATA QUALITY REPORT")
    print("="*60)
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✅ No missing values detected")
    else:
        print(f"⚠️  Missing values found:\n{missing[missing > 0]}")
    
    # Feature correlations with target
    print("\n🔗 Top Features Correlated with Irrigation Amount:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = df[numeric_cols].corr()['irrigation_amount'].sort_values(ascending=False)
    print(correlations.head(10).to_string())
    
    # Distribution info
    print(f"\n📊 Irrigation Amount Distribution:")
    print(f"Mean: {df['irrigation_amount'].mean():.2f} L/ha")
    print(f"Median: {df['irrigation_amount'].median():.2f} L/ha")
    print(f"Std Dev: {df['irrigation_amount'].std():.2f} L/ha")
    print(f"Range: [{df['irrigation_amount'].min():.2f}, {df['irrigation_amount'].max():.2f}]")


def main():
    """Run complete data generation pipeline"""
    
    print("\n" + "="*60)
    print("🚀 IRRIGATION DATA GENERATION PIPELINE")
    print("="*60 + "\n")
    
    # Generate base data
    df = generate_irrigation_data(n_samples=10000, seed=42)
    
    # Engineer features
    df = engineer_features(df)
    
    # Save data
    raw_path, processed_path = split_and_save(df)
    
    # Print quality report
    visualize_data(df)
    
    print("\n" + "="*60)
    print("✅ DATA GENERATION COMPLETED")
    print("="*60)
    print(f"\n📁 Files created:")
    print(f"   - Raw data: {raw_path}")
    print(f"   - Processed data: {processed_path}")
    print(f"\n🎯 Next step: Run training pipeline")
    print(f"   python src/training/train_pipeline.py\n")


if __name__ == "__main__":
    main()