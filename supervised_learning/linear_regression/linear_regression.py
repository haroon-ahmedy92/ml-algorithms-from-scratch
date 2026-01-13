"""
Linear Regression from Scratch
==============================

This module implements Linear Regression using two methods:
1. Normal Equation (Closed-form solution)
2. Gradient Descent (Iterative optimization)

What is Linear Regression?
--------------------------
Linear Regression is a supervised learning algorithm used to predict a continuous
target variable (y) based on one or more input features (X). It assumes a linear
relationship between the inputs and the output:

    y = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ

Or in matrix form:
    y = Xw

Where:
    - X: Input features (m samples × n features)
    - w: Weights/coefficients (n features × 1)
    - y: Target values (m samples × 1)

When to Use Linear Regression:
------------------------------
✓ Predicting continuous values (e.g., house prices, temperatures, sales)
✓ When you expect a linear relationship between features and target
✓ When interpretability is important (coefficients show feature importance)
✓ As a baseline model before trying complex algorithms

Strengths:
----------
+ Simple and fast to train
+ Highly interpretable (each weight shows feature impact)
+ Works well when the true relationship is approximately linear
+ No hyperparameters needed for Normal Equation
+ Low computational cost for predictions

Limitations:
------------
- Assumes linear relationship (can't capture complex patterns)
- Sensitive to outliers
- Assumes features are independent (multicollinearity issues)
- Normal Equation is slow for large datasets (matrix inversion is O(n³))

Author: ML Algorithms from Scratch Project
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    """
    Linear Regression model with two training methods.
    
    Attributes:
        weights (np.ndarray): Model weights including bias term (w₀, w₁, ..., wₙ)
        method (str): Training method used ('normal_equation' or 'gradient_descent')
        learning_rate (float): Learning rate for gradient descent
        n_iterations (int): Number of iterations for gradient descent
        cost_history (list): History of cost values during gradient descent training
    """
    
    def __init__(self, method='normal_equation', learning_rate=0.01, n_iterations=1000):
        """
        Initialize the Linear Regression model.
        
        Parameters:
            method (str): Training method - 'normal_equation' or 'gradient_descent'
            learning_rate (float): Learning rate for gradient descent (default: 0.01)
            n_iterations (int): Number of iterations for gradient descent (default: 1000)
        """
        self.weights = None
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.cost_history = []
    
    def _add_bias_term(self, X):
        """
        Add a column of ones to X for the bias term (intercept).
        
        This transforms X from [x₁, x₂, ...] to [1, x₁, x₂, ...]
        so we can include the bias w₀ in the weight vector.
        
        Parameters:
            X (np.ndarray): Feature matrix (m samples × n features)
            
        Returns:
            np.ndarray: Feature matrix with bias column (m samples × (n+1) features)
        """
        m = X.shape[0]  # Number of samples
        ones = np.ones((m, 1))  # Column of ones for bias
        return np.concatenate([ones, X], axis=1)
    
    def fit_normal_equation(self, X, y):
        """
        Train using the Normal Equation (closed-form solution).
        
        The Normal Equation gives the optimal weights directly:
        
            w = (XᵀX)⁻¹ Xᵀy
        
        Derivation:
        1. We want to minimize the cost function: J(w) = (1/2m) ||Xw - y||²
        2. Take derivative with respect to w: ∂J/∂w = (1/m) Xᵀ(Xw - y)
        3. Set derivative to zero: Xᵀ(Xw - y) = 0
        4. Solve for w: XᵀXw = Xᵀy → w = (XᵀX)⁻¹ Xᵀy
        
        Time Complexity: O(n³) due to matrix inversion
        
        Parameters:
            X (np.ndarray): Feature matrix (m samples × n features)
            y (np.ndarray): Target values (m samples,)
        """
        # Add bias term to X
        X_b = self._add_bias_term(X)
        
        # Normal Equation: w = (XᵀX)⁻¹ Xᵀy
        # Using np.linalg.pinv for numerical stability (handles singular matrices)
        XtX = np.dot(X_b.T, X_b)  # XᵀX
        XtX_inv = np.linalg.pinv(XtX)  # (XᵀX)⁻¹
        Xty = np.dot(X_b.T, y)  # Xᵀy
        
        self.weights = np.dot(XtX_inv, Xty)  # Final weights
    
    def fit_gradient_descent(self, X, y):
        """
        Train using Gradient Descent (iterative optimization).
        
        Gradient Descent iteratively updates weights to minimize the cost:
        
            w := w - α * ∂J/∂w
        
        Where:
        - α is the learning rate
        - ∂J/∂w = (1/m) Xᵀ(Xw - y) is the gradient
        
        The cost function (Mean Squared Error):
            J(w) = (1/2m) Σ(ŷᵢ - yᵢ)²
        
        Parameters:
            X (np.ndarray): Feature matrix (m samples × n features)
            y (np.ndarray): Target values (m samples,)
        """
        # Add bias term to X
        X_b = self._add_bias_term(X)
        m, n = X_b.shape  # m = samples, n = features (including bias)
        
        # Initialize weights to zeros
        self.weights = np.zeros(n)
        self.cost_history = []
        
        # Gradient Descent iterations
        for i in range(self.n_iterations):
            # Step 1: Make predictions with current weights
            predictions = np.dot(X_b, self.weights)  # ŷ = Xw
            
            # Step 2: Calculate error
            errors = predictions - y  # (ŷ - y)
            
            # Step 3: Calculate gradient
            # ∂J/∂w = (1/m) Xᵀ(ŷ - y)
            gradient = (1/m) * np.dot(X_b.T, errors)
            
            # Step 4: Update weights
            # w := w - α * gradient
            self.weights = self.weights - self.learning_rate * gradient
            
            # Step 5: Calculate and store cost for monitoring
            cost = (1/(2*m)) * np.sum(errors**2)
            self.cost_history.append(cost)
    
    def fit(self, X, y):
        """
        Train the model using the specified method.
        
        Parameters:
            X (np.ndarray): Feature matrix (m samples × n features)
            y (np.ndarray): Target values (m samples,)
            
        Returns:
            self: Returns the instance for method chaining
        """
        # Ensure y is 1D array
        y = np.array(y).flatten()
        X = np.array(X)
        
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if self.method == 'normal_equation':
            self.fit_normal_equation(X, y)
        elif self.method == 'gradient_descent':
            self.fit_gradient_descent(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}. "
                           "Use 'normal_equation' or 'gradient_descent'")
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters:
            X (np.ndarray): Feature matrix (m samples × n features)
            
        Returns:
            np.ndarray: Predicted values (m samples,)
        """
        if self.weights is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X_b = self._add_bias_term(X)
        return np.dot(X_b, self.weights)
    
    def score(self, X, y):
        """
        Calculate R² score (coefficient of determination).
        
        R² = 1 - (SS_res / SS_tot)
        
        Where:
        - SS_res = Σ(yᵢ - ŷᵢ)² (residual sum of squares)
        - SS_tot = Σ(yᵢ - ȳ)² (total sum of squares)
        
        R² ranges from -∞ to 1:
        - R² = 1: Perfect predictions
        - R² = 0: Model predicts the mean
        - R² < 0: Model is worse than predicting the mean
        
        Parameters:
            X (np.ndarray): Feature matrix
            y (np.ndarray): True target values
            
        Returns:
            float: R² score
        """
        y = np.array(y).flatten()
        y_pred = self.predict(X)
        
        ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
        
        return 1 - (ss_res / ss_tot)
    
    def get_coefficients(self):
        """
        Get the model coefficients (weights).
        
        Returns:
            dict: Dictionary with 'bias' and 'weights' keys
        """
        if self.weights is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        return {
            'bias': self.weights[0],
            'weights': self.weights[1:]
        }
    
    def plot_cost_history(self):
        """
        Plot the cost function over iterations (only for gradient descent).
        
        This visualization helps diagnose training:
        - Decreasing curve: Learning is working
        - Flat curve: May have converged or learning rate too small
        - Increasing/oscillating: Learning rate too high
        """
        if not self.cost_history:
            print("No cost history available. Train with gradient descent first.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history, 'b-', linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Cost (MSE)', fontsize=12)
        plt.title('Gradient Descent: Cost vs Iterations', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_regression_line(X, y, model, title="Linear Regression"):
    """
    Visualize the regression line with data points (for 1D features).
    
    Parameters:
        X (np.ndarray): Feature values
        y (np.ndarray): Target values
        model (LinearRegression): Trained model
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(X, y, color='blue', alpha=0.6, label='Data points', s=50)
    
    # Plot regression line
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, color='red', linewidth=2, label='Regression line')
    
    # Formatting
    plt.xlabel('X (Feature)', fontsize=12)
    plt.ylabel('y (Target)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# EXAMPLE USAGE AND DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LINEAR REGRESSION FROM SCRATCH - DEMONSTRATION")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Create a sample dataset
    # -------------------------------------------------------------------------
    # We'll create synthetic data following: y = 4 + 3x + noise
    # True parameters: bias = 4, weight = 3
    
    np.random.seed(42)  # For reproducibility
    
    # Generate 100 random samples
    X = 2 * np.random.rand(100, 1)  # Features between 0 and 2
    y = 4 + 3 * X.flatten() + np.random.randn(100) * 0.5  # y = 4 + 3x + noise
    
    print("\n📊 Dataset Information:")
    print(f"   - Number of samples: {len(X)}")
    print(f"   - True relationship: y = 4 + 3x + noise")
    print(f"   - Feature range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"   - Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # -------------------------------------------------------------------------
    # Method 1: Normal Equation
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("📐 METHOD 1: NORMAL EQUATION")
    print("-" * 60)
    
    model_ne = LinearRegression(method='normal_equation')
    model_ne.fit(X, y)
    
    coeffs_ne = model_ne.get_coefficients()
    r2_ne = model_ne.score(X, y)
    
    print(f"\n   Learned Parameters:")
    print(f"   - Bias (intercept): {coeffs_ne['bias']:.4f} (true: 4.0)")
    print(f"   - Weight (slope):   {coeffs_ne['weights'][0]:.4f} (true: 3.0)")
    print(f"\n   Model Performance:")
    print(f"   - R² Score: {r2_ne:.4f}")
    
    # -------------------------------------------------------------------------
    # Method 2: Gradient Descent
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("📉 METHOD 2: GRADIENT DESCENT")
    print("-" * 60)
    
    model_gd = LinearRegression(
        method='gradient_descent',
        learning_rate=0.1,
        n_iterations=1000
    )
    model_gd.fit(X, y)
    
    coeffs_gd = model_gd.get_coefficients()
    r2_gd = model_gd.score(X, y)
    
    print(f"\n   Hyperparameters:")
    print(f"   - Learning rate: {model_gd.learning_rate}")
    print(f"   - Iterations: {model_gd.n_iterations}")
    print(f"\n   Learned Parameters:")
    print(f"   - Bias (intercept): {coeffs_gd['bias']:.4f} (true: 4.0)")
    print(f"   - Weight (slope):   {coeffs_gd['weights'][0]:.4f} (true: 3.0)")
    print(f"\n   Model Performance:")
    print(f"   - R² Score: {r2_gd:.4f}")
    print(f"   - Final Cost: {model_gd.cost_history[-1]:.6f}")
    
    # -------------------------------------------------------------------------
    # Compare Results
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("🔍 COMPARISON OF METHODS")
    print("-" * 60)
    print(f"\n   {'Metric':<20} {'Normal Equation':<18} {'Gradient Descent':<18}")
    print(f"   {'-'*20} {'-'*18} {'-'*18}")
    print(f"   {'Bias':<20} {coeffs_ne['bias']:<18.4f} {coeffs_gd['bias']:<18.4f}")
    print(f"   {'Weight':<20} {coeffs_ne['weights'][0]:<18.4f} {coeffs_gd['weights'][0]:<18.4f}")
    print(f"   {'R² Score':<20} {r2_ne:<18.4f} {r2_gd:<18.4f}")
    
    # -------------------------------------------------------------------------
    # Make Predictions
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("🎯 SAMPLE PREDICTIONS")
    print("-" * 60)
    
    test_values = np.array([[0.5], [1.0], [1.5]])
    predictions = model_ne.predict(test_values)
    true_values = 4 + 3 * test_values.flatten()  # Without noise
    
    print(f"\n   {'X Value':<12} {'Predicted':<12} {'True (no noise)':<15}")
    print(f"   {'-'*12} {'-'*12} {'-'*15}")
    for x, pred, true in zip(test_values.flatten(), predictions, true_values):
        print(f"   {x:<12.2f} {pred:<12.4f} {true:<15.4f}")
    
    # -------------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("📈 VISUALIZATIONS")
    print("-" * 60)
    print("\n   Generating plots...")
    
    # Plot 1: Regression line with Normal Equation
    plot_regression_line(X, y, model_ne, 
                        title="Linear Regression - Normal Equation")
    
    # Plot 2: Regression line with Gradient Descent
    plot_regression_line(X, y, model_gd, 
                        title="Linear Regression - Gradient Descent")
    
    # Plot 3: Cost history for Gradient Descent
    model_gd.plot_cost_history()
    
    print("\n" + "=" * 60)
    print("✅ DEMONSTRATION COMPLETE!")
    print("=" * 60)
