# Logistic Regression from Scratch

Implementation of **Logistic Regression** following the Coursera Machine Learning Specialization (Andrew Ng) notation and methodology.

## Overview

Logistic Regression is a supervised learning algorithm used for **binary classification** problems. Despite its name, it's a classification algorithm that predicts the probability of an instance belonging to a particular class.

## Key Concepts

### Sigmoid Function
The sigmoid (logistic) function maps any real-valued number to a value between 0 and 1:

$$g(z) = \frac{1}{1 + e^{-z}}$$

### Hypothesis Function
For logistic regression, the hypothesis is:

$$f_{w,b}(x) = g(w \cdot x + b) = \frac{1}{1 + e^{-(w \cdot x + b)}}$$

Where:
- $w$ = weight vector (parameters)
- $b$ = bias term (parameter)
- $x$ = input features
- $g(z)$ = sigmoid function

### Cost Function
The cost function for logistic regression (binary cross-entropy loss):

$$J(w,b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(f_{w,b}(x^{(i)})) + (1 - y^{(i)}) \log(1 - f_{w,b}(x^{(i)})) \right]$$

### Gradient Descent
Update rules for parameters:

$$w_j := w_j - \alpha \frac{\partial J(w,b)}{\partial w_j}$$
$$b := b - \alpha \frac{\partial J(w,b)}{\partial b}$$

Where:
- $\alpha$ = learning rate
- The gradients are:
  - $\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)}) x_j^{(i)}$
  - $\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)}) - y^{(i)})$

## Features

- ✅ **Vectorized implementation** using NumPy for efficiency
- ✅ **Supports both univariate and multivariate** classification
- ✅ **Binary cross-entropy cost function**
- ✅ **Gradient descent optimization**
- ✅ **Probability predictions** (`predict_proba`)
- ✅ **Class predictions** with customizable threshold
- ✅ **Accuracy scoring**
- ✅ **Visualization tools**:
  - Sigmoid function plot
  - Decision boundary (for 2D features)
  - Cost history over iterations
  - 1D classification visualization

## Usage

### Basic Example

```python
import numpy as np
from logistic_regression import LogisticRegression

# Generate sample data
np.random.seed(42)
m = 100

# Class 0: centered around (2, 2)
X_class0 = np.random.randn(m // 2, 2) * 0.8 + [2, 2]
y_class0 = np.zeros(m // 2)

# Class 1: centered around (5, 5)
X_class1 = np.random.randn(m // 2, 2) * 0.8 + [5, 5]
y_class1 = np.ones(m // 2)

X_train = np.vstack([X_class0, X_class1])
y_train = np.hstack([y_class0, y_class1])

# Create and train model
model = LogisticRegression(alpha=0.1, num_iters=1000)
model.fit(X_train, y_train)

# Make predictions
X_test = np.array([[3, 3], [4, 4]])
probabilities = model.predict_proba(X_test)  # Get probabilities
predictions = model.predict(X_test)          # Get class labels (0 or 1)

# Evaluate
accuracy = model.score(X_train, y_train)
print(f"Accuracy: {accuracy:.2%}")
```

### Visualization

```python
from logistic_regression import plot_decision_boundary, plot_sigmoid

# Plot sigmoid function
plot_sigmoid()

# Plot decision boundary (requires 2D data)
plot_decision_boundary(X_train, y_train, model)

# Plot cost history
model.plot_cost_history()
```

### Using Standalone Functions

```python
from logistic_regression import sigmoid, compute_cost, compute_gradient, gradient_descent

# Sigmoid function
z = np.array([-2, -1, 0, 1, 2])
g_z = sigmoid(z)

# Compute cost
w = np.array([0.5, 0.5])
b = 0.0
cost = compute_cost(X_train, y_train, w, b)

# Compute gradients
dj_dw, dj_db = compute_gradient(X_train, y_train, w, b)

# Run gradient descent
w_init = np.zeros(2)
b_init = 0.0
alpha = 0.1
num_iters = 1000

w_final, b_final, J_history = gradient_descent(
    X_train, y_train, w_init, b_init, alpha, num_iters,
    compute_cost, compute_gradient
)
```

## API Reference

### LogisticRegression Class

#### Constructor
```python
LogisticRegression(alpha=0.01, num_iters=1000)
```
- `alpha` (float): Learning rate
- `num_iters` (int): Number of gradient descent iterations

#### Methods

**`fit(X_train, y_train, w_init=None, b_init=0.0)`**
- Train the model using gradient descent
- `X_train`: Training features, shape (m, n) or (m,)
- `y_train`: Binary labels (0 or 1), shape (m,)
- `w_init`: Initial weights (default: zeros)
- `b_init`: Initial bias (default: 0.0)
- Returns: self

**`predict_proba(X)`**
- Predict probabilities for input samples
- `X`: Features, shape (m, n) or (m,)
- Returns: Probabilities, shape (m,)

**`predict(X, threshold=0.5)`**
- Predict class labels for input samples
- `X`: Features, shape (m, n) or (m,)
- `threshold`: Decision threshold (default: 0.5)
- Returns: Class labels (0 or 1), shape (m,)

**`score(X, y)`**
- Calculate accuracy on given data
- Returns: Accuracy (float between 0 and 1)

**`get_parameters()`**
- Get learned parameters
- Returns: Dict with keys 'w' and 'b'

**`plot_cost_history()`**
- Plot cost function over iterations

### Standalone Functions

**`sigmoid(z)`**
- Compute sigmoid function: $g(z) = \frac{1}{1 + e^{-z}}$

**`compute_cost(X, y, w, b)`**
- Compute binary cross-entropy cost

**`compute_gradient(X, y, w, b)`**
- Compute gradients of cost function

**`gradient_descent(X, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function)`**
- Run gradient descent optimization

**`plot_decision_boundary(X, y, model, title)`**
- Visualize decision boundary (requires 2D features)

**`plot_sigmoid()`**
- Plot the sigmoid activation function

## Running the Demo

Execute the module directly to see comprehensive demonstrations:

```bash
python logistic_regression.py
```

This will:
1. Generate synthetic binary classification datasets
2. Demonstrate the sigmoid function
3. Train models on 2D and 1D data
4. Display predictions and accuracy
5. Create visualizations:
   - Sigmoid function plot
   - Decision boundary plot
   - Cost convergence plot
   - 1D classification visualization

## Understanding the Output

### Training Output
```
Training Logistic Regression Model:
  Number of training examples (m): 100
  Number of features (n): 2
  Learning rate (alpha): 0.1
  Number of iterations: 1000

Running gradient descent...

Iteration    0: Cost 0.693147
Iteration  100: Cost 0.284562
Iteration  200: Cost 0.198341
...
Training complete!
  Final parameters:
    w = [0.8234 0.8234]
    b = -5.1234
  Final cost: 0.123456
```

### Decision Threshold
By default, predictions use a threshold of 0.5:
- If $f_{w,b}(x) \geq 0.5$ → predict class 1
- If $f_{w,b}(x) < 0.5$ → predict class 0

You can customize this:
```python
predictions = model.predict(X_test, threshold=0.7)
```

## When to Use Logistic Regression

**Good for:**
- Binary classification problems
- When you need probability estimates
- Linearly separable classes
- Interpretable models (feature importance via weights)
- Baseline model for classification tasks

**Not ideal for:**
- Non-linear decision boundaries (consider feature engineering or other algorithms)
- Multi-class classification (use one-vs-rest or multinomial logistic regression)
- Very high-dimensional data without regularization

## Extensions and Improvements

This implementation can be extended with:
1. **Regularization** (L1/L2) to prevent overfitting
2. **Feature scaling** for faster convergence
3. **Mini-batch or stochastic gradient descent** for large datasets
4. **Multi-class classification** (one-vs-rest or softmax)
5. **Polynomial features** for non-linear boundaries
6. **Early stopping** based on validation performance

## Mathematical Intuition

### Why Sigmoid?
- Maps any real number to (0, 1) → interpretable as probability
- Smooth and differentiable → works well with gradient descent
- S-shaped curve → natural decision boundary at 0.5

### Why Cross-Entropy Loss?
- Derived from maximum likelihood estimation
- Convex function → single global minimum
- Heavily penalizes confident wrong predictions
- Gradient has elegant form: $(f_{w,b}(x) - y) \cdot x$

### Gradient Descent Convergence
- Learning rate too high → oscillations or divergence
- Learning rate too low → slow convergence
- Proper learning rate → smooth, steady decrease in cost

## Comparison with Linear Regression

| Aspect | Linear Regression | Logistic Regression |
|--------|------------------|---------------------|
| **Use case** | Regression (continuous output) | Classification (discrete output) |
| **Output** | Any real number | Probability (0 to 1) |
| **Hypothesis** | $f(x) = w \cdot x + b$ | $f(x) = \frac{1}{1 + e^{-(w \cdot x + b)}}$ |
| **Cost function** | Mean Squared Error | Binary Cross-Entropy |
| **Decision** | N/A | Threshold probability |

## Dependencies

- NumPy: Numerical computations
- Matplotlib: Visualization

## License

MIT License - Feel free to use and modify!

## Author

ML Algorithms from Scratch Project - Following Coursera ML Specialization (Andrew Ng)
