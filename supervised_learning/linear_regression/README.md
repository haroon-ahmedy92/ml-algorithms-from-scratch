# Linear Regression

This module contains a Linear Regression implementation from scratch using only NumPy and Matplotlib.

## What is Linear Regression?

Linear Regression is a supervised learning algorithm used to predict a continuous target variable based on one or more input features. It assumes a linear relationship between inputs and outputs:

```
y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
```

## When to Use It

- Predicting continuous values (house prices, temperatures, sales)
- When you expect a linear relationship between features and target
- When interpretability is important
- As a baseline before trying complex algorithms

## Strengths

- Simple and fast to train
- Highly interpretable coefficients
- Low computational cost for predictions
- No hyperparameters needed for Normal Equation

## Limitations

- Assumes linear relationship
- Sensitive to outliers
- Multicollinearity issues with correlated features
- Normal Equation is O(n³) for matrix inversion

## Files

- `linear_regression.py` - Main implementation with both training methods

## Usage

```python
from linear_regression import LinearRegression

# Method 1: Normal Equation
model = LinearRegression(method='normal_equation')
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Method 2: Gradient Descent
model = LinearRegression(
    method='gradient_descent',
    learning_rate=0.01,
    n_iterations=1000
)
model.fit(X_train, y_train)
```

## Run Demo

```bash
python linear_regression.py
```
