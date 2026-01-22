# Neurons and Layers - TensorFlow/Keras Demonstration

A comprehensive Python script demonstrating the inner workings of **Neurons and Layers** using TensorFlow/Keras, recreating the "Neurons and Layers" lab from the Coursera Machine Learning Specialization.

## Overview

This educational script provides hands-on demonstrations of:
1. **Linear Regression** - A neuron without activation (linear activation)
2. **Logistic Regression** - A neuron with sigmoid activation

Each section includes detailed mathematical explanations and compares Keras predictions with manual NumPy calculations to verify correctness.

## Installation

### Method 1: Virtual Environment (Recommended)

This is the recommended approach for modern Python development and avoids system package conflicts.

```bash
# Navigate to the neural_networks directory
cd supervised_learning/neural_networks

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

**Note:** Remember to activate the virtual environment each time you want to run the script:
```bash
source venv/bin/activate
```

### Method 2: System Packages (Debian/Ubuntu/Parrot OS)

For Debian-based systems with externally-managed Python environments:

```bash
# Install system packages via apt
sudo apt install python3-numpy python3-matplotlib python3-tensorflow python3-full

# No activation needed, run scripts directly with python3
```

### Method 3: Using --break-system-packages (Not Recommended)

Only use this if you understand the risks:

```bash
pip3 install --break-system-packages -r requirements.txt
```

⚠️ **Warning:** This may cause conflicts with system packages.

## Running the Script

### With Virtual Environment

```bash
# Activate virtual environment (if not already active)
source venv/bin/activate

# Run the demonstration
python neurons_and_layers_demo.py

# When finished, deactivate
deactivate
```

### With System Packages

```bash
# Run directly with python3
python3 neurons_and_layers_demo.py
```

### Interactive Mode

The script runs interactively with pause points between sections:
- Press Enter to proceed through each section
- View visualizations as they appear
- Read detailed console output explaining each step

## What the Script Demonstrates

### Section 1: Linear Regression (Neuron without Activation)

**Mathematical Formula:**
$$f_{w,b}(x) = wx + b$$

**What it does:**
- Creates a single Dense layer with `linear` activation
- Manually sets weights: $w = 200$, $b = 100$
- Makes predictions using Keras
- Performs manual calculations using NumPy
- Compares results to verify they match
- Visualizes the regression line

**Example Output:**
```
Input: x = 1.0
Keras prediction: f(1.0) = 300.0
Manual calculation: 200 * 1.0 + 100 = 300.0
✓ Match!
```

**Code Snippet:**
```python
# Create Dense layer with linear activation
linear_layer = Dense(units=1, activation='linear', input_dim=1)

# Set weights manually
linear_layer.set_weights([np.array([[200.0]]), np.array([100.0])])

# Make predictions
predictions = linear_layer(X_train)
```

### Section 2: Logistic Regression (Neuron with Sigmoid Activation)

**Mathematical Formula:**
$$f_{w,b}(x) = g(wx + b)$$

where the sigmoid function is:
$$g(z) = \frac{1}{1 + e^{-z}}$$

Combined:
$$f_{w,b}(x) = \frac{1}{1 + e^{-(wx + b)}}$$

**What it does:**
- Creates a Sequential model with sigmoid activation
- Manually sets weights: $w = 2.0$, $b = -4.5$
- Demonstrates probability predictions (0 to 1)
- Shows detailed step-by-step calculation for first data point
- Compares Keras vs manual sigmoid calculations
- Visualizes decision boundary and sigmoid curve

**Example Output:**
```
Input: x = 0.0
Step 1: z = 2.0 * 0.0 + (-4.5) = -4.5
Step 2: g(z) = 1 / (1 + e^(4.5)) = 0.011
Keras prediction: 0.010889
Manual calculation: 0.010889
✓ Match!
```

**Code Snippet:**
```python
# Create Sequential model with sigmoid activation
model = Sequential([
    Dense(units=1, activation='sigmoid', input_shape=(1,))
])

# Set weights manually
model.set_weights([np.array([[2.0]]), np.array([-4.5])])

# Make predictions
probabilities = model.predict(X_train)
```

## Features

✅ **Detailed Mathematical Explanations** - LaTeX formulas in comments  
✅ **Step-by-Step Calculations** - Shows each computational step  
✅ **Verification** - Compares Keras with manual NumPy calculations  
✅ **Visualizations** - Matplotlib plots for regression lines and sigmoid curves  
✅ **Interactive Learning** - Pause points to absorb information  
✅ **Educational Comments** - Extensive documentation throughout  
✅ **Real Examples** - House price prediction and exam pass/fail scenarios  

## Understanding the Output

### Console Output Structure

Each section provides:

1. **Training Data Display**
   - Shows input features and target values
   - Explains data shapes and dimensions

2. **Model Architecture**
   - Details of Dense layers
   - Activation functions used
   - Weight and bias parameters

3. **Predictions**
   - Keras model predictions
   - Manual NumPy calculations
   - Side-by-side comparison

4. **Verification**
   - Match confirmation (✓/✗)
   - Numerical differences (if any)

5. **Visualizations**
   - Regression lines
   - Sigmoid curves
   - Decision boundaries

### Example Console Output

```
================================================================================
SECTION 1: LINEAR REGRESSION - NEURON WITHOUT ACTIVATION
================================================================================

📊 Step 1: Define Training Data
--------------------------------------------------------------------------------
Input features (X_train):
[[1.]
 [2.]]
Shape: (2, 1)

Target values (Y_train):
[[300.]
 [500.]]
Shape: (2, 1)

🧠 Step 2: Create Keras Model with Linear Activation
--------------------------------------------------------------------------------
✓ Created Dense layer:
  - Units: 1 (single neuron)
  - Activation: linear (f(x) = x)
  - Input dimension: 1

⚙️  Step 3: Set Weights and Bias Manually
--------------------------------------------------------------------------------
Set weights manually:
  w = 200.0 (weight)
  b = 100.0 (bias)

Mathematical formula: f(x) = 200.0x + 100.0
```

## Visualizations

### 1. Linear Regression Plot
- Red X marks: Training data
- Blue circles: Keras predictions
- Green line: Regression line $f(x) = 200x + 100$

### 2. Sigmoid Function Plot
- Shows the characteristic S-curve
- Decision threshold at 0.5
- Full range from 0 to 1

### 3. Logistic Regression Plot
- Red/Blue markers: Training data (class 0/1)
- Green curve: Sigmoid probability curve
- Purple line: Decision boundary
- Orange line: 0.5 threshold

## Key Concepts Explained

### What is a Neuron?

A neuron is a computational unit that:
1. **Receives inputs** (features $x$)
2. **Applies weights** (parameters $w$)
3. **Adds bias** (parameter $b$)
4. **Applies activation function** (linear, sigmoid, ReLU, etc.)
5. **Produces output** (prediction or probability)

### Activation Functions

| Function | Formula | Use Case | Output Range |
|----------|---------|----------|--------------|
| **Linear** | $f(z) = z$ | Regression | $(-\infty, \infty)$ |
| **Sigmoid** | $f(z) = \frac{1}{1+e^{-z}}$ | Binary classification | $(0, 1)$ |
| ReLU | $f(z) = \max(0, z)$ | Hidden layers | $[0, \infty)$ |
| Tanh | $f(z) = \tanh(z)$ | Hidden layers | $(-1, 1)$ |

### Dense Layer in Keras

A **Dense** (fully connected) layer:
- Connects every input to every neuron
- Implements: `output = activation(dot(input, weights) + bias)`
- Most common layer type in neural networks

**Parameters:**
- `units`: Number of neurons in the layer
- `activation`: Activation function to use
- `input_shape` or `input_dim`: Shape of input data

### Why Sigmoid for Binary Classification?

1. **Probability Interpretation**: Output is between 0 and 1
2. **Smooth Gradient**: Differentiable everywhere (good for training)
3. **Decision Boundary**: Natural threshold at 0.5
4. **Mathematical Properties**: Related to logistic function and odds ratio

## Comparison: Linear vs Sigmoid Activation

| Aspect | Linear Activation | Sigmoid Activation |
|--------|------------------|-------------------|
| **Formula** | $f(x) = x$ | $f(x) = \frac{1}{1+e^{-x}}$ |
| **Output Range** | $(-\infty, \infty)$ | $(0, 1)$ |
| **Use Case** | Regression | Binary Classification |
| **Interpretation** | Continuous value | Probability |
| **Decision Rule** | N/A | Threshold at 0.5 |

## Educational Value

This script is perfect for:

✓ **Understanding Neural Network Basics** - Start with single neurons  
✓ **Learning TensorFlow/Keras API** - Practical hands-on examples  
✓ **Verifying Mathematical Understanding** - Manual calculations match theory  
✓ **Debugging ML Models** - See exact computations  
✓ **Teaching Material** - Comprehensive comments and explanations  

## Extending the Script

### Add More Activation Functions

```python
# ReLU activation
relu_layer = Dense(units=1, activation='relu', input_dim=1)

# Tanh activation
tanh_layer = Dense(units=1, activation='tanh', input_dim=1)

# Softmax for multi-class (requires multiple units)
softmax_layer = Dense(units=3, activation='softmax', input_dim=2)
```

### Multiple Neurons

```python
# Layer with 5 neurons
multi_neuron_layer = Dense(units=5, activation='relu', input_dim=3)

# Each neuron has its own weights and bias
# Total parameters: (3 inputs × 5 neurons) + 5 biases = 20 parameters
```

### Multi-Layer Networks

```python
# Build a deeper network
model = Sequential([
    Dense(units=4, activation='relu', input_shape=(2,)),   # Hidden layer
    Dense(units=4, activation='relu'),                     # Hidden layer
    Dense(units=1, activation='sigmoid')                   # Output layer
])
```

## Common Issues and Solutions

### Issue: TensorFlow Not Installed

```bash
# Error: ModuleNotFoundError: No module named 'tensorflow'

# Solution:
pip3 install tensorflow
```

### Issue: Matplotlib Display Problems

```bash
# If plots don't display, you may need:
pip3 install python3-tk

# Or use non-interactive backend:
import matplotlib
matplotlib.use('Agg')
```

### Issue: NumPy Array Shape Mismatches

```python
# Always check shapes
print(f"Shape: {X_train.shape}")

# Reshape if needed
X_train = X_train.reshape(-1, 1)  # Convert to column vector
```

## Mathematical Derivations

### Why These Gradients?

For **linear regression**, the gradient of the cost function with respect to weights is:

$$\frac{\partial J}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)}) \cdot x^{(i)}$$

For **logistic regression**, the gradient has the same form:

$$\frac{\partial J}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)}) \cdot x^{(i)}$$

Even though the cost functions are different (MSE vs cross-entropy), the gradient computation has the same structure!

### Decision Boundary Calculation

For sigmoid activation with $f(x) = \frac{1}{1 + e^{-(wx+b)}}$:

Decision boundary is where $f(x) = 0.5$:
- $\frac{1}{1 + e^{-(wx+b)}} = 0.5$
- $1 + e^{-(wx+b)} = 2$
- $e^{-(wx+b)} = 1$
- $-(wx+b) = 0$
- $wx + b = 0$
- $x = -\frac{b}{w}$

In our example: $x = -\frac{(-4.5)}{2.0} = 2.25$ hours

## Dependencies

- **NumPy** (≥1.19.0): Numerical computations
- **Matplotlib** (≥3.3.0): Visualizations
- **TensorFlow** (≥2.0.0): Neural network framework

## License

MIT License - Free to use and modify for educational purposes!

## Author

ML Algorithms from Scratch Project  
Following Coursera Machine Learning Specialization (Andrew Ng)  
Date: January 22, 2026

## References

- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
- [Coursera ML Specialization](https://www.coursera.org/specializations/machine-learning-introduction)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)

---

**Happy Learning! 🧠🚀**
