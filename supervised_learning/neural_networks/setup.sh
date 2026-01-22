#!/bin/bash
# Setup script for Neurons and Layers demonstration
# This script creates a virtual environment and installs dependencies

set -e  # Exit on error

echo "=========================================="
echo "Neurons and Layers - Setup Script"
echo "=========================================="
echo ""

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed"
    echo "   Please install python3 first:"
    echo "   sudo apt install python3 python3-full python3-venv"
    exit 1
fi

echo "✓ Python3 found: $(python3 --version)"
echo ""

# Check if we're in the correct directory
if [ ! -f "neurons_and_layers_demo.py" ]; then
    echo "❌ Error: neurons_and_layers_demo.py not found in current directory"
    echo "   Please run this script from the neural_networks directory:"
    echo "   cd supervised_learning/neural_networks"
    echo "   bash setup.sh"
    exit 1
fi

echo "✓ Found neurons_and_layers_demo.py"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "   Virtual environment already exists. Removing old one..."
    rm -rf venv
fi

python3 -m venv venv

echo "✓ Virtual environment created"
echo ""

# Activate virtual environment and install dependencies
echo "📥 Installing dependencies..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip > /dev/null 2>&1

# Install requirements
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✓ Dependencies installed from requirements.txt"
else
    echo "⚠️  Warning: requirements.txt not found"
    echo "   Installing packages manually..."
    pip install numpy matplotlib tensorflow
    echo "✓ Dependencies installed manually"
fi

echo ""

# Verify installation
echo "🔍 Verifying installation..."
python -c "import numpy; print(f'  ✓ NumPy {numpy.__version__}')"
python -c "import matplotlib; print(f'  ✓ Matplotlib {matplotlib.__version__}')"
python -c "import tensorflow as tf; print(f'  ✓ TensorFlow {tf.__version__}')"

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "To run the demonstration:"
echo ""
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run the script:"
echo "     python neurons_and_layers_demo.py"
echo ""
echo "  3. When finished, deactivate:"
echo "     deactivate"
echo ""
echo "=========================================="
