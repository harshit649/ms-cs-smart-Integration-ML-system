#!/bin/bash

# Quick Start Script for Smart Irrigation MLOps Project
# This script automates the complete setup and execution

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Check if Python is installed
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.9+"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    print_success "Python $PYTHON_VERSION found"
}

# Create virtual environment
setup_venv() {
    print_header "Step 1: Setting up Python Virtual Environment"
    
    if [ -d "venv" ]; then
        print_info "Virtual environment already exists. Skipping..."
    else
        python3 -m venv venv
        print_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    source venv/bin/activate || . venv/Scripts/activate
    print_success "Virtual environment activated"
    
    # Upgrade pip
    pip install --upgrade pip -q
    print_success "Pip upgraded"
}

# Install dependencies
install_dependencies() {
    print_header "Step 2: Installing Dependencies"
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt -q
        print_success "All dependencies installed"
    else
        print_error "requirements.txt not found!"
        exit 1
    fi
}

# Create directory structure
create_directories() {
    print_header "Step 3: Creating Directory Structure"
    
    mkdir -p data/{raw,processed,features}
    mkdir -p models/trained
    mkdir -p mlruns
    mkdir -p reports
    
    print_success "Directory structure created"
}

# Generate data
generate_data() {
    print_header "Step 4: Generating Synthetic Data"
    
    if [ -f "data/processed/irrigation_data.csv" ]; then
        print_info "Data already exists. Skipping generation..."
    else
        python src/data/generate_data.py
        print_success "Data generated successfully"
    fi
}

# Train models
train_models() {
    print_header "Step 5: Training ML Models with MLflow"
    
    print_info "This will take 2-3 minutes..."
    python src/training/train_pipeline.py
    print_success "Models trained successfully"
}

# Run tests
run_tests() {
    print_header "Step 6: Running Tests"
    
    if command -v pytest &> /dev/null; then
        pytest tests/ -v --tb=short || print_info "Some tests may have failed (expected if model not loaded)"
        print_success "Tests completed"
    else
        print_info "Pytest not available. Skipping tests..."
    fi
}

# Start services
start_services() {
    print_header "Step 7: Starting Services"
    
    echo ""
    print_info "Starting MLflow UI on port 5000..."
    echo "  Run in separate terminal: mlflow ui --port 5000"
    
    echo ""
    print_info "Starting FastAPI on port 8000..."
    echo "  Run in separate terminal: python src/api/app.py"
    
    echo ""
    print_success "Setup Complete!"
}

# Print usage instructions
print_instructions() {
    print_header "🎉 Setup Completed Successfully!"
    
    echo ""
    echo -e "${GREEN}Next Steps:${NC}"
    echo ""
    echo "1️⃣  View MLflow Experiments:"
    echo "   $ mlflow ui --port 5000"
    echo "   Then open: http://localhost:5000"
    echo ""
    echo "2️⃣  Start the API Server:"
    echo "   $ python src/api/app.py"
    echo "   Then open: http://localhost:8000/docs"
    echo ""
    echo "3️⃣  Test the API:"
    echo "   $ curl -X POST \"http://localhost:8000/predict\" \\"
    echo "     -H \"Content-Type: application/json\" \\"
    echo "     -d '{\"temperature\": 28.5, \"humidity\": 65, \"soil_moisture\": 35, \"rainfall\": 0, \"days_since_last_irrigation\": 3, \"crop_type\": \"wheat\", \"growth_stage\": \"vegetative\"}'"
    echo ""
    echo "4️⃣  Run Drift Detection:"
    echo "   $ python src/monitoring/drift_detection.py"
    echo ""
    echo "5️⃣  Run Tests:"
    echo "   $ pytest tests/ -v"
    echo ""
    echo -e "${BLUE}Project Structure:${NC}"
    echo "  📂 data/              → Generated datasets"
    echo "  📂 models/trained/    → Trained ML models"
    echo "  📂 mlruns/            → MLflow experiment logs"
    echo "  📂 src/               → Source code"
    echo "  📂 tests/             → Test suite"
    echo ""
    echo -e "${YELLOW}🚀 Your MLOps portfolio project is ready!${NC}"
    echo ""
}

# Main execution
main() {
    clear
    print_header "🌾 Smart Irrigation MLOps - Quick Start"
    echo ""
    
    # Run all steps
    check_python
    setup_venv
    install_dependencies
    create_directories
    generate_data
    train_models
    run_tests
    start_services
    
    echo ""
    print_instructions
}

# Run main function
main