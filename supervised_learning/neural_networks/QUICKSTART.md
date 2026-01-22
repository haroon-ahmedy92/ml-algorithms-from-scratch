# Quick Start Guide - Neurons and Layers Demo

## Installation (One-Time Setup)

```bash
# Navigate to the neural_networks directory
cd supervised_learning/neural_networks

# Install dependencies
pip3 install -r requirements.txt
```

## Running the Script

```bash
# Run the complete demonstration
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

**TensorFlow not found?**
```bash
pip3 install tensorflow
```

**Display issues?**
```bash
# Install display backend
sudo apt-get install python3-tk
```

## Next Steps

After understanding this demo, you can:
- Explore multi-layer networks
- Try different activation functions (ReLU, tanh)
- Build custom architectures
- Train models on real datasets

---
**Happy Learning! 🧠**
