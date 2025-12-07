#!/bin/bash

# Setup and Test Script
# This script sets up the environment and runs basic tests

echo "=========================================="
echo "Antibiotic Resistance ML - Setup & Test"
echo "=========================================="

# Check Python version
echo ""
echo "Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "
import pandas as pd
import numpy as np
import sklearn
import xgboost
import streamlit
print('✓ All core packages installed successfully')
print(f'  - pandas: {pd.__version__}')
print(f'  - numpy: {np.__version__}')
print(f'  - scikit-learn: {sklearn.__version__}')
print(f'  - xgboost: {xgboost.__version__}')
print(f'  - streamlit: {streamlit.__version__}')
"

# Check data
echo ""
echo "Checking data files..."
python3 -c "
from pathlib import Path
import pandas as pd

raw_data = Path('data/raw/raw_data.csv')
if raw_data.exists():
    df = pd.read_csv(raw_data)
    print(f'✓ Raw data found: {len(df)} rows, {len(df.columns)} columns')
else:
    print('✗ Raw data not found')
"

# Directory structure
echo ""
echo "Verifying directory structure..."
python3 -c "
from pathlib import Path

dirs = [
    'data/raw',
    'data/processed',
    'data/results',
    'src/preprocessing',
    'src/classification',
    'src/clustering',
    'src/association_rules',
    'src/dimensionality_reduction',
    'src/statistical_analysis',
    'src/deployment'
]

all_exist = True
for d in dirs:
    path = Path(d)
    exists = path.exists()
    print(f'  {d}: {\"✓\" if exists else \"✗\"}')
    if not exists:
        all_exist = False

if all_exist:
    print('\n✓ All directories present')
else:
    print('\n⚠ Some directories missing')
"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate venv: source venv/bin/activate"
echo "2. Run pipeline: python run_pipeline.py"
echo "3. Launch dashboard: streamlit run src/deployment/app.py"
echo ""
