#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print step header
print_step() {
    echo -e "\n${YELLOW}=== $1 ===${NC}\n"
}

# Function to check if previous command was successful
check_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Success${NC}"
    else
        echo -e "${RED}✗ Failed${NC}"
        exit 1
    fi
}

# Create necessary directories
print_step "Creating directories"
mkdir -p data/{raw,processed,models} logs
check_status

# Setup Python virtual environment
print_step "Setting up Python virtual environment"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
check_status

# Download dataset
print_step "Downloading dataset"
python src/data/download_dataset.py --dataset bot-iot --use-mirror
check_status

# Preprocess data
print_step "Preprocessing data"
python src/data/preprocess.py
check_status

# Train model
print_step "Training model"
python src/models/train.py --mode advanced
check_status

# Run tests
print_step "Running tests"
pytest tests/
check_status

# Evaluate model
print_step "Evaluating model performance"
python src/utils/evaluate.py \
    --test_data data/processed/features_test.parquet \
    --model_path data/models/model.onnx \
    --batch_size 32
check_status

# Start detection system
print_step "Starting detection system"
echo -e "${GREEN}Starting detection system...${NC}"
echo -e "Access the web interface at ${YELLOW}http://localhost:8080${NC}"
echo -e "Press Ctrl+C to stop the system"
python docs/website/server.py 