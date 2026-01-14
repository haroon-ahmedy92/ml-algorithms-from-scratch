"""
Linear Regression from Scratch
==============================

This module implements Linear Regression following the Coursera Machine Learning 
Specialization (Andrew Ng) notation and structure.

Notation:
---------
    - x_train: Input features (training data)
    - y_train: Target values (training data)
    - m: Number of training examples
    - w: Weight parameter (scalar for univariate, vector for multivariate)
    - b: Bias parameter (scalar)
    - alpha: Learning rate
    - f_wb: Prediction f_w,b(x) = wx + b
    - dj_dw: Gradient of cost function with respect to w
    - dj_db: Gradient of cost function with respect to b

Cost Function:
--------------
    J(w,b) = (1/2m) * Σ(f_w,b(x^(i)) - y^(i))²
    
where f_w,b(x^(i)) = w * x^(i) + b

Implementation:
---------------
This implementation uses vectorized NumPy operations for efficient computation.

Author: ML Algorithms from Scratch Project
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_cost(x, y, w, b):
    """
    Compute cost function for linear regression using vectorized operations.
    
    J(w,b) = (1/2m) * Σ(f_w,b(x^(i)) - y^(i))²
    
    Parameters:
        x (ndarray): Shape (m,) - Input features
        y (ndarray): Shape (m,) - Target values
        w (scalar): Weight parameter
        b (scalar): Bias parameter
        
    Returns:
        total_cost (scalar): The cost J(w,b)
    """
    m = x.shape[0]
    
    # Vectorized prediction: f_wb for all examples at once
    f_wb = w * x + b
    
    # Vectorized cost calculation
    total_cost = np.sum((f_wb - y) ** 2) / (2 * m)
    
    return total_cost


def compute_gradient(x, y, w, b):
    """
    Compute gradient for linear regression using vectorized operations.
    
    ∂J/∂w = (1/m) * Σ(f_w,b(x^(i)) - y^(i)) * x^(i)
    ∂J/∂b = (1/m) * Σ(f_w,b(x^(i)) - y^(i))
    
    Parameters:
        x (ndarray): Shape (m,) - Input features
        y (ndarray): Shape (m,) - Target values
        w (scalar): Weight parameter
        b (scalar): Bias parameter
        
    Returns:
        dj_dw (scalar): Gradient of cost with respect to w
        dj_db (scalar): Gradient of cost with respect to b
    """
    m = x.shape[0]
    
    # Vectorized prediction for all examples
    f_wb = w * x + b
    
    # Vectorized error calculation
    error = f_wb - y
    
    # Vectorized gradient calculations
    dj_dw = np.dot(error, x) / m
    dj_db = np.sum(error) / m
    
    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
    Perform gradient descent to learn w and b.
    
    Updates w and b by taking num_iters gradient steps with learning rate alpha.
    
    Parameters:
        x (ndarray): Shape (m,) - Input features
        y (ndarray): Shape (m,) - Target values
        w_in (scalar): Initial weight parameter
        b_in (scalar): Initial bias parameter
        alpha (scalar): Learning rate
        num_iters (int): Number of iterations to run gradient descent
        cost_function: Function to compute cost
        gradient_function: Function to compute gradients
        
    Returns:
        w (scalar): Updated weight parameter after gradient descent
        b (scalar): Updated bias parameter after gradient descent
        J_history (list): History of cost values
    """
    J_history = []
    w = w_in
    b = b_in
    
    for i in range(num_iters):
        # Calculate gradients
        dj_dw, dj_db = gradient_function(x, y, w, b)
        
        # Update parameters simultaneously
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        # Save cost J at each iteration
        if i < 100000:  # Prevent resource exhaustion
            cost = cost_function(x, y, w, b)
            J_history.append(cost)
        
        # Print cost every 10% of iterations
        if i % (num_iters // 10) == 0 or i == num_iters - 1:
            cost = cost_function(x, y, w, b)
            print(f"Iteration {i:4d}: Cost {cost:8.2f}")
    
    return w, b, J_history


class LinearRegression:
    """
    Linear Regression model following Coursera ML Specialization notation.
    
    Attributes:
        w (scalar): Weight parameter
        b (scalar): Bias parameter
        alpha (float): Learning rate
        num_iters (int): Number of iterations for gradient descent
        J_history (list): History of cost values during training
    """
    
    def __init__(self, alpha=0.01, num_iters=1000):
        """
        Initialize the Linear Regression model.
        
        Parameters:
            alpha (float): Learning rate (default: 0.01)
            num_iters (int): Number of iterations for gradient descent (default: 1000)
        """
        self.w = None
        self.b = None
        self.alpha = alpha
        self.num_iters = num_iters
        self.J_history = []
    
    def fit(self, x_train, y_train, w_init=0.0, b_init=0.0):
        """
        Train the model using gradient descent.
        
        Parameters:
            x_train (ndarray): Shape (m,) - Training features
            y_train (ndarray): Shape (m,) - Training targets
            w_init (scalar): Initial weight parameter (default: 0.0)
            b_init (scalar): Initial bias parameter (default: 0.0)
            
        Returns:
            self: Returns the instance for method chaining
        """
        # Ensure inputs are numpy arrays and 1D
        x_train = np.array(x_train).flatten()
        y_train = np.array(y_train).flatten()
        
        m = x_train.shape[0]
        
        print(f"\nTraining Linear Regression Model:")
        print(f"  Number of training examples (m): {m}")
        print(f"  Learning rate (alpha): {self.alpha}")
        print(f"  Number of iterations: {self.num_iters}")
        print(f"\nRunning gradient descent...\n")
        
        # Run gradient descent
        self.w, self.b, self.J_history = gradient_descent(
            x_train, y_train, w_init, b_init, 
            self.alpha, self.num_iters, 
            compute_cost, compute_gradient
        )
        
        print(f"\nTraining complete!")
        print(f"  Final parameters: w = {self.w:.4f}, b = {self.b:.4f}")
        print(f"  Final cost: {self.J_history[-1]:.6f}")
        
        return self
    
    def predict(self, x):
        """
        Make predictions using the trained model.
        
        f_w,b(x) = w * x + b
        
        Parameters:
            x (ndarray or scalar): Input feature(s)
            
        Returns:
            ndarray or scalar: Predicted value(s)
        """
        if self.w is None or self.b is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        # Handle both scalar and array inputs
        if np.isscalar(x):
            return self.w * x + self.b
        
        x = np.array(x).flatten()
        return self.w * x + self.b
    
    def compute_cost(self, x, y):
        """
        Compute the cost for the current parameters.
        
        Parameters:
            x (ndarray): Shape (m,) - Input features
            y (ndarray): Shape (m,) - Target values
            
        Returns:
            scalar: The cost J(w,b)
        """
        if self.w is None or self.b is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        return compute_cost(x, y, self.w, self.b)
    
    def score(self, x, y):
        """
        Calculate R² score (coefficient of determination).
        
        R² = 1 - (SS_res / SS_tot)
        
        Parameters:
            x (ndarray): Feature values
            y (ndarray): True target values
            
        Returns:
            float: R² score
        """
        y = np.array(y).flatten()
        y_pred = self.predict(x)
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        return 1 - (ss_res / ss_tot)
    
    def get_parameters(self):
        """
        Get the model parameters.
        
        Returns:
            dict: Dictionary with 'w' and 'b' keys
        """
        if self.w is None or self.b is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        return {
            'w': self.w,
            'b': self.b
        }
    
    def plot_cost_history(self):
        """
        Plot the cost function J(w,b) over iterations.
        
        Helps diagnose training:
        - Decreasing curve: Learning is working
        - Flat curve: May have converged or alpha too small
        - Increasing/oscillating: alpha too high
        """
        if not self.J_history:
            print("No cost history available. Train the model first.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.J_history, 'b-', linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Cost J(w,b)', fontsize=12)
        plt.title(f'Gradient Descent: Cost vs Iterations (alpha={self.alpha})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_regression_line(x, y, model, title="Linear Regression"):
    """
    Visualize the regression line with data points.
    
    Parameters:
        x (ndarray): Feature values
        y (ndarray): Target values
        model (LinearRegression): Trained model
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Plot training data
    plt.scatter(x, y, color='blue', alpha=0.6, marker='x', s=100, label='Training data')
    
    # Plot regression line
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, color='red', linewidth=2, 
             label=f'f_w,b(x) = {model.w:.2f}x + {model.b:.2f}')
    
    # Formatting
    plt.xlabel('x (Feature)', fontsize=12)
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
    print("=" * 70)
    print("LINEAR REGRESSION - COURSERA ML SPECIALIZATION IMPLEMENTATION")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Create training dataset
    # -------------------------------------------------------------------------
    # Synthetic data: y = 4 + 3x + noise
    
    np.random.seed(42)
    
    # Generate m training examples
    m = 100
    x_train = 2 * np.random.rand(m)
    y_train = 4 + 3 * x_train + np.random.randn(m) * 0.5
    
    print(f"\n📊 Dataset Information:")
    print(f"   Number of training examples (m): {m}")
    print(f"   True relationship: y = 4 + 3x + noise")
    print(f"   Feature range: [{x_train.min():.2f}, {x_train.max():.2f}]")
    print(f"   Target range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    
    # -------------------------------------------------------------------------
    # Test the standalone functions
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("🔬 TESTING STANDALONE FUNCTIONS")
    print("-" * 70)
    
    # Initialize parameters
    w_test = 0.0
    b_test = 0.0
    
    # Test compute_cost
    initial_cost = compute_cost(x_train, y_train, w_test, b_test)
    print(f"\nInitial cost with w={w_test}, b={b_test}: {initial_cost:.4f}")
    
    # Test compute_gradient
    dj_dw, dj_db = compute_gradient(x_train, y_train, w_test, b_test)
    print(f"Initial gradients:")
    print(f"  dj_dw = {dj_dw:.4f}")
    print(f"  dj_db = {dj_db:.4f}")
    
    # -------------------------------------------------------------------------
    # Train model using gradient descent
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("🎓 TRAINING MODEL WITH GRADIENT DESCENT")
    print("-" * 70)
    
    model = LinearRegression(alpha=0.01, num_iters=1000)
    model.fit(x_train, y_train, w_init=0.0, b_init=0.0)
    
    # Get learned parameters
    params = model.get_parameters()
    r2_score = model.score(x_train, y_train)
    
    print(f"\n📈 Model Performance:")
    print(f"   R² Score: {r2_score:.4f}")
    
    # -------------------------------------------------------------------------
    # Make predictions
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("🎯 SAMPLE PREDICTIONS")
    print("-" * 70)
    
    x_test = np.array([0.5, 1.0, 1.5])
    y_pred = model.predict(x_test)
    y_true = 4 + 3 * x_test  # True values without noise
    
    print(f"\n   {'x':<10} {'Predicted':<15} {'True (no noise)':<15}")
    print(f"   {'-'*10} {'-'*15} {'-'*15}")
    for x_val, pred, true in zip(x_test, y_pred, y_true):
        print(f"   {x_val:<10.2f} {pred:<15.4f} {true:<15.4f}")
    
    # -------------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("📊 GENERATING VISUALIZATIONS")
    print("-" * 70)
    print("\nPlotting regression line and cost history...")
    
    # Plot 1: Regression line
    plot_regression_line(x_train, y_train, model, 
                        title="Linear Regression - Coursera ML Specialization")
    
    # Plot 2: Cost history
    model.plot_cost_history()
    
    print("\n" + "=" * 70)
    print("✅ DEMONSTRATION COMPLETE!")
    print("=" * 70)

