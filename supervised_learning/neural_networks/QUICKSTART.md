# Quick Start Guide - Neurons and Layers Demo

## Installation (One-Time Setup)

### Option 1: Using Virtual Environment (Recommended)

```bash
# Navigate to the neural_networks directory
cd supervised_learning/neural_networks

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using System Packages (Debian/Ubuntu)

```bash
# Install TensorFlow and dependencies via apt
sudo apt install python3-numpy python3-matplotlib python3-tensorflow
```

### Option 3: Using pipx (For standalone applications)

```bash
# Install pipx if not already installed
sudo apt install pipx

# Note: This is better for command-line tools, not recommended for this script
```

## Running the Script

### If using Virtual Environment

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run the complete demonstration
python neurons_and_layers_demo.py

# When done, deactivate the virtual environment
deactivate
```

### If using System Packages

```bash
# Run directly with python3
python3 neurons_and_layers_demo.py
```

## What You'll See

### Part 1: Linear Regression
- Training data for house price prediction
- Keras Dense layer with linear activation
- Manual weight setting (w=200, b=100)
- Prediction comparison (Keras vs NumPy)
- Visualization of regression line

### Part 2: Logistic Regression  
- Training data for binary classification (pass/fail)
- Keras Sequential model with sigmoid activation
- Manual weight setting (w=2, b=-4.5)
- Step-by-step calculation walkthrough
- Decision boundary visualization

## Sample Output

```
================================================================================
SECTION 1: LINEAR REGRESSION - NEURON WITHOUT ACTIVATION
================================================================================

📊 Step 1: Define Training Data
Input: x = 1.0 → Prediction: 300.0
Input: x = 2.0 → Prediction: 500.0

✅ Keras predictions match manual calculations!

================================================================================
SECTION 2: LOGISTIC REGRESSION - NEURON WITH SIGMOID ACTIVATION
================================================================================

🎯 Detailed Calculation for X[0] = 0:
  z = 2.0 * 0.0 + (-4.5) = -4.5
  sigmoid(-4.5) = 0.010889
  
✅ Keras predictions match manual calculations!
```

## Key Learning Points

1. **Understanding Neurons**: See how a single neuron computes outputs
2. **Activation Functions**: Compare linear vs sigmoid behavior
3. **Keras API**: Learn Dense layers and Sequential models
4. **Verification**: All predictions verified against manual math
5. **Visualization**: Interactive plots showing decision boundaries

## Troubleshooting

**Externally-managed-environment error?**
```bash
# Use a virtual environment instead (see Installation above)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**TensorFlow not found (in virtual environment)?**
```bash
# Activate virtual environment first
source venv/bin/activate
pip install tensorflow
```

**TensorFlow not found (system-wide)?**
```bash
# Install via apt on Debian/Ubuntu
sudo apt install python3-tensorflow
```

**Display issues?**
```bash
# Install display backend
sudo apt-get install python3-tk python3-matplotlib
```

**Virtual environment activation issues?**
```bash
# Make sure you're in the right directory
cd supervised_learning/neural_networks

# Check if venv exists
ls -la venv/

# If not, create it
python3 -m venv venv
```

## Next Steps

After understanding this demo, you can:
- Explore multi-layer networks
- Try different activation functions (ReLU, tanh)
- Build custom architectures
- Train models on real datasets

---
**Happy Learning! 🧠**
