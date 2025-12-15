"""
Data drift detection and model monitoring system.
Tracks feature distributions and model performance over time.
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from datetime import datetime
import json
import os
from typing import Dict, List, Tuple


class DriftDetector:
    """Detect data drift in production predictions"""
    
    def __init__(self, reference_data_path: str, threshold: float = 0.05):
        """
        Initialize drift detector
        
        Args:
            reference_data_path: Path to training/reference dataset
            threshold: P-value threshold for KS test (default: 0.05)
        """
        self.threshold = threshold
        self.reference_data = pd.read_csv(reference_data_path)
        
        # Numeric features to monitor
        self.numeric_features = [
            'temperature', 'humidity', 'soil_moisture', 'rainfall',
            'days_since_last_irrigation', 'evapotranspiration'
        ]
        
        # Calculate reference statistics
        self.reference_stats = self._calculate_statistics(self.reference_data)
        
        print(f"✅ Drift detector initialized with {len(self.reference_data)} reference samples")
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate statistical metrics for dataset"""
        stats = {}
        for feature in self.numeric_features:
            if feature in df.columns:
                stats[feature] = {
                    'mean': df[feature].mean(),
                    'std': df[feature].std(),
                    'min': df[feature].min(),
                    'max': df[feature].max(),
                    'median': df[feature].median(),
                    'q25': df[feature].quantile(0.25),
                    'q75': df[feature].quantile(0.75)
                }
        return stats
    
    def detect_drift(self, production_data: pd.DataFrame) -> Dict:
        """
        Detect drift between reference and production data
        
        Args:
            production_data: Recent production predictions
        
        Returns:
            Dictionary with drift detection results
        """
        print("\n" + "="*60)
        print("🔍 DRIFT DETECTION ANALYSIS")
        print("="*60)
        
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'n_reference_samples': len(self.reference_data),
            'n_production_samples': len(production_data),
            'features_analyzed': len(self.numeric_features),
            'drift_detected': False,
            'drifted_features': [],
            'feature_details': {}
        }
        
        # Run KS test for each feature
        for feature in self.numeric_features:
            if feature not in production_data.columns:
                print(f"⚠️  Feature '{feature}' not found in production data")
                continue
            
            # Kolmogorov-Smirnov test
            reference_values = self.reference_data[feature].dropna()
            production_values = production_data[feature].dropna()
            
            ks_statistic, p_value = ks_2samp(reference_values, production_values)
            
            # Check if drift detected
            is_drifted = p_value < self.threshold
            
            # Calculate statistics for production data
            prod_stats = {
                'mean': production_values.mean(),
                'std': production_values.std(),
                'median': production_values.median()
            }
            
            ref_stats = self.reference_stats[feature]
            
            # Calculate percentage changes
            mean_change_pct = ((prod_stats['mean'] - ref_stats['mean']) / ref_stats['mean']) * 100
            std_change_pct = ((prod_stats['std'] - ref_stats['std']) / ref_stats['std']) * 100
            
            # Store results
            drift_results['feature_details'][feature] = {
                'ks_statistic': float(ks_statistic),
                'p_value': float(p_value),
                'drift_detected': is_drifted,
                'reference_mean': float(ref_stats['mean']),
                'production_mean': float(prod_stats['mean']),
                'mean_change_pct': float(mean_change_pct),
                'reference_std': float(ref_stats['std']),
                'production_std': float(prod_stats['std']),
                'std_change_pct': float(std_change_pct)
            }
            
            # Print results
            status = "🚨 DRIFT" if is_drifted else "✅ OK"
            print(f"\n{status} {feature}:")
            print(f"  KS Statistic: {ks_statistic:.4f}")
            print(f"  P-value: {p_value:.4f}")
            print(f"  Mean: {ref_stats['mean']:.2f} → {prod_stats['mean']:.2f} ({mean_change_pct:+.1f}%)")
            print(f"  Std: {ref_stats['std']:.2f} → {prod_stats['std']:.2f} ({std_change_pct:+.1f}%)")
            
            if is_drifted:
                drift_results['drift_detected'] = True
                drift_results['drifted_features'].append(feature)
        
        # Summary
        print("\n" + "="*60)
        if drift_results['drift_detected']:
            print(f"⚠️  DRIFT DETECTED in {len(drift_results['drifted_features'])} feature(s):")
            for feature in drift_results['drifted_features']:
                print(f"   - {feature}")
            print("\n💡 Recommendation: Consider model retraining")
        else:
            print("✅ No significant drift detected")
            print("   Model performance expected to be stable")
        print("="*60)
        
        return drift_results
    
    def monitor_predictions(self, predictions: np.ndarray, threshold_degradation: float = 0.1):
        """
        Monitor prediction distribution for anomalies
        
        Args:
            predictions: Array of model predictions
            threshold_degradation: Acceptable degradation threshold
        """
        print("\n📊 Prediction Monitoring:")
        
        pred_mean = predictions.mean()
        pred_std = predictions.std()
        pred_min = predictions.min()
        pred_max = predictions.max()
        
        # Check for anomalous predictions
        anomalies = (predictions < 0) | (predictions > 1000)  # Outside reasonable range
        n_anomalies = anomalies.sum()
        
        print(f"  Mean prediction: {pred_mean:.2f} L/ha")
        print(f"  Std deviation: {pred_std:.2f}")
        print(f"  Range: [{pred_min:.2f}, {pred_max:.2f}]")
        
        if n_anomalies > 0:
            print(f"  ⚠️  {n_anomalies} anomalous predictions detected")
        else:
            print(f"  ✅ All predictions within expected range")
        
        return {
            'mean': float(pred_mean),
            'std': float(pred_std),
            'min': float(pred_min),
            'max': float(pred_max),
            'n_anomalies': int(n_anomalies),
            'anomaly_rate': float(n_anomalies / len(predictions))
        }
    
    def save_report(self, results: Dict, output_path: str = "reports/drift_report.json"):
        """Save drift detection report"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Report saved: {output_path}")


def simulate_production_data(reference_data: pd.DataFrame, 
                             n_samples: int = 1000,
                             inject_drift: bool = False) -> pd.DataFrame:
    """
    Simulate production data for testing
    
    Args:
        reference_data: Training dataset
        n_samples: Number of samples to generate
        inject_drift: Whether to inject artificial drift
    """
    # Sample from reference data
    prod_data = reference_data.sample(n_samples, replace=True).reset_index(drop=True)
    
    if inject_drift:
        print("⚠️  Injecting artificial drift for testing...")
        # Shift temperature distribution
        prod_data['temperature'] = prod_data['temperature'] + np.random.normal(3, 1, n_samples)
        # Shift humidity distribution
        prod_data['humidity'] = prod_data['humidity'] - np.random.normal(5, 2, n_samples)
        # Add noise to soil moisture
        prod_data['soil_moisture'] = prod_data['soil_moisture'] + np.random.normal(0, 5, n_samples)
    
    return prod_data


def main():
    """Run drift detection demo"""
    
    print("\n" + "="*60)
    print("🚀 MODEL MONITORING & DRIFT DETECTION DEMO")
    print("="*60)
    
    # Load reference data
    reference_path = "data/processed/irrigation_data.csv"
    
    if not os.path.exists(reference_path):
        print(f"❌ Reference data not found: {reference_path}")
        print("   Please run data generation first: python src/data/generate_data.py")
        return
    
    reference_data = pd.read_csv(reference_path)
    
    # Initialize detector
    detector = DriftDetector(reference_path, threshold=0.05)
    
    # Test 1: No drift scenario
    print("\n\n🧪 TEST 1: Production data WITHOUT drift")
    print("-" * 60)
    prod_data_normal = simulate_production_data(reference_data, n_samples=500, inject_drift=False)
    results_normal = detector.detect_drift(prod_data_normal)
    
    # Test 2: Drift scenario
    print("\n\n🧪 TEST 2: Production data WITH drift")
    print("-" * 60)
    prod_data_drift = simulate_production_data(reference_data, n_samples=500, inject_drift=True)
    results_drift = detector.detect_drift(prod_data_drift)
    
    # Save reports
    os.makedirs("reports", exist_ok=True)
    detector.save_report(results_normal, "reports/drift_report_normal.json")
    detector.save_report(results_drift, "reports/drift_report_drift.json")
    
    print("\n" + "="*60)
    print("✅ MONITORING DEMO COMPLETED")
    print("="*60)
    print("\n💡 In production, run this periodically to monitor model health")
    print("   Example: python src/monitoring/drift_detection.py --threshold 0.05\n")


if __name__ == "__main__":
    main()