"""
Unit tests for FastAPI irrigation prediction service.
Demonstrates testing best practices for ML APIs.
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.app import app, IrrigationRequest


# Initialize test client
client = TestClient(app)


class TestHealthEndpoints:
    """Test health check and info endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns correct info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert data["service"] == "Smart Irrigation ML API"
        assert "version" in data
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data


class TestPredictionEndpoint:
    """Test prediction endpoint functionality"""
    
    @pytest.fixture
    def valid_request(self):
        """Sample valid request payload"""
        return {
            "temperature": 28.5,
            "humidity": 65.0,
            "soil_moisture": 35.2,
            "rainfall": 0.0,
            "days_since_last_irrigation": 3,
            "crop_type": "wheat",
            "growth_stage": "vegetative",
            "evapotranspiration": 5.2
        }
    
    def test_valid_prediction(self, valid_request):
        """Test prediction with valid input"""
        response = client.post("/predict", json=valid_request)
        
        # May return 503 if model not loaded, which is acceptable in testing
        if response.status_code == 503:
            pytest.skip("Model not loaded - expected in CI/CD environment")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "irrigation_amount" in data
        assert "unit" in data
        assert "confidence_interval" in data
        assert "model_version" in data
        assert "timestamp" in data
        assert "recommendation" in data
        
        # Check data types and values
        assert isinstance(data["irrigation_amount"], float)
        assert data["irrigation_amount"] >= 0
        assert data["unit"] == "liters/hectare"
        assert len(data["confidence_interval"]) == 2
    
    def test_prediction_with_optional_et(self):
        """Test prediction without providing ET (should be calculated)"""
        request = {
            "temperature": 30.0,
            "humidity": 60.0,
            "soil_moisture": 40.0,
            "rainfall": 5.0,
            "days_since_last_irrigation": 2,
            "crop_type": "corn",
            "growth_stage": "flowering"
            # No evapotranspiration provided
        }
        
        response = client.post("/predict", json=request)
        
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        assert response.status_code == 200
        data = response.json()
        assert "irrigation_amount" in data


class TestInputValidation:
    """Test input validation and error handling"""
    
    def test_invalid_temperature(self):
        """Test with out-of-range temperature"""
        request = {
            "temperature": 60.0,  # Too high
            "humidity": 65.0,
            "soil_moisture": 35.0,
            "rainfall": 0.0,
            "days_since_last_irrigation": 3,
            "crop_type": "wheat",
            "growth_stage": "vegetative"
        }
        
        response = client.post("/predict", json=request)
        assert response.status_code == 422  # Validation error
    
    def test_invalid_humidity(self):
        """Test with out-of-range humidity"""
        request = {
            "temperature": 25.0,
            "humidity": 150.0,  # Too high
            "soil_moisture": 35.0,
            "rainfall": 0.0,
            "days_since_last_irrigation": 3,
            "crop_type": "wheat",
            "growth_stage": "vegetative"
        }
        
        response = client.post("/predict", json=request)
        assert response.status_code == 422
    
    def test_invalid_crop_type(self):
        """Test with invalid crop type"""
        request = {
            "temperature": 25.0,
            "humidity": 65.0,
            "soil_moisture": 35.0,
            "rainfall": 0.0,
            "days_since_last_irrigation": 3,
            "crop_type": "banana",  # Not in allowed list
            "growth_stage": "vegetative"
        }
        
        response = client.post("/predict", json=request)
        assert response.status_code == 422
    
    def test_missing_required_field(self):
        """Test with missing required field"""
        request = {
            "temperature": 25.0,
            "humidity": 65.0,
            # Missing soil_moisture
            "rainfall": 0.0,
            "days_since_last_irrigation": 3,
            "crop_type": "wheat",
            "growth_stage": "vegetative"
        }
        
        response = client.post("/predict", json=request)
        assert response.status_code == 422
    
    def test_negative_rainfall(self):
        """Test that negative rainfall is rejected"""
        request = {
            "temperature": 25.0,
            "humidity": 65.0,
            "soil_moisture": 35.0,
            "rainfall": -5.0,  # Negative not allowed
            "days_since_last_irrigation": 3,
            "crop_type": "wheat",
            "growth_stage": "vegetative"
        }
        
        response = client.post("/predict", json=request)
        assert response.status_code == 422


class TestBatchPrediction:
    """Test batch prediction endpoint"""
    
    def test_batch_prediction(self):
        """Test batch prediction with multiple requests"""
        batch_requests = [
            {
                "temperature": 28.5,
                "humidity": 65.0,
                "soil_moisture": 35.0,
                "rainfall": 0.0,
                "days_since_last_irrigation": 3,
                "crop_type": "wheat",
                "growth_stage": "vegetative"
            },
            {
                "temperature": 30.0,
                "humidity": 60.0,
                "soil_moisture": 40.0,
                "rainfall": 5.0,
                "days_since_last_irrigation": 2,
                "crop_type": "corn",
                "growth_stage": "flowering"
            }
        ]
        
        response = client.post("/batch_predict", json=batch_requests)
        
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "count" in data
        assert data["count"] == 2
        assert len(data["predictions"]) == 2
    
    def test_batch_size_limit(self):
        """Test that batch size is limited"""
        # Create 101 requests (over limit of 100)
        large_batch = [{
            "temperature": 25.0,
            "humidity": 65.0,
            "soil_moisture": 35.0,
            "rainfall": 0.0,
            "days_since_last_irrigation": 3,
            "crop_type": "wheat",
            "growth_stage": "vegetative"
        }] * 101
        
        response = client.post("/batch_predict", json=large_batch)
        assert response.status_code == 400


class TestModelInfo:
    """Test model information endpoint"""
    
    def test_model_info(self):
        """Test model info endpoint"""
        response = client.get("/model/info")
        
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        assert response.status_code == 200
        data = response.json()
        assert "model_version" in data
        assert "model_type" in data
        assert "n_features" in data
        assert "features" in data


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_zero_values(self):
        """Test with zero values for various fields"""
        request = {
            "temperature": 20.0,
            "humidity": 50.0,
            "soil_moisture": 30.0,
            "rainfall": 0.0,  # Zero rainfall
            "days_since_last_irrigation": 0,  # Just irrigated
            "crop_type": "wheat",
            "growth_stage": "vegetative",
            "evapotranspiration": 0.0  # Minimal ET
        }
        
        response = client.post("/predict", json=request)
        
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        assert response.status_code == 200
    
    def test_maximum_values(self):
        """Test with maximum allowed values"""
        request = {
            "temperature": 50.0,  # Max temperature
            "humidity": 100.0,  # Max humidity
            "soil_moisture": 100.0,  # Max soil moisture
            "rainfall": 500.0,  # Max rainfall
            "days_since_last_irrigation": 30,  # Max days
            "crop_type": "rice",
            "growth_stage": "maturity",
            "evapotranspiration": 20.0  # Max ET
        }
        
        response = client.post("/predict", json=request)
        
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        assert response.status_code == 200
        data = response.json()
        # With such extreme conditions, irrigation should still be reasonable
        assert data["irrigation_amount"] < 1000


# Pytest configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])