"""
Linear Regression with Feature Scaling (Z-Score Normalization)
===============================================================

This module implements Linear Regression following the Coursera Machine Learning 
Specialization (Andrew Ng) notation with Z-Score Normalization for faster convergence.

Notation:
---------
    - X_train: Training example matrix (m rows, n columns)
    - y_train: Training example targets (vector of length m)
    - X[i], y[i]: The i-th training example and target
    - m: Number of training examples
    - n: Number of features in each example
    - w: Parameter: weights (vector of length n)
    - b: Parameter: bias (scalar)
    - alpha: Learning rate
    - f_wb: Model prediction f_w,b(X) = w·X + b
    - J(w,b): Cost function
    - dj_dw: Gradient of cost function with respect to w
    - dj_db: Gradient of cost function with respect to b

Z-Score Normalization (Feature Scaling):
-----------------------------------------
    To speed up gradient descent, features are scaled so they have a mean (μ) of 0
    and a standard deviation (σ) of 1.
    
    x_norm^(i)_j = (x^(i)_j - μ_j) / σ_j
    
    Where:
    - μ_j = mean of feature j
    - σ_j = standard deviation of feature j

Cost Function:
--------------
    J(w,b) = (1/2m) * Σ(f_w,b(X^(i)) - y^(i))²
    
where f_w,b(X^(i)) = w·X^(i) + b

Implementation:
---------------
This implementation uses vectorized NumPy operations for efficient computation.
Supports both univariate and multivariate linear regression with feature scaling.

Author: ML Algorithms from Scratch Project
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt


def normalize_features(X_train, mu=None, sigma=None):
    """
    Normalize features using Z-Score Normalization.
    
    x_norm_j = (x_j - μ_j) / σ_j
    
    Parameters:
        X_train (ndarray): Shape (m, n) - Training feature matrix
        mu (ndarray): Shape (n,) - Mean of each feature (computed if None)
        sigma (ndarray): Shape (n,) - Std dev of each feature (computed if None)
        
    Returns:
        X_norm (ndarray): Shape (m, n) - Normalized features
        mu (ndarray): Mean values used for normalization
        sigma (ndarray): Standard deviation values used for normalization
    """
    # Compute mean and std if not provided (for training data)
    if mu is None:
        mu = np.mean(X_train, axis=0)
    if sigma is None:
        sigma = np.std(X_train, axis=0)
    
    # Normalize: subtract mean and divide by standard deviation
    X_norm = (X_train - mu) / sigma
    
    return X_norm, mu, sigma


def compute_cost(X, y, w, b):
    """
    Compute cost function for linear regression using vectorized operations.
    
    J(w,b) = (1/2m) * Σ(f_w,b(X^(i)) - y^(i))²
    
    Parameters:
        X (ndarray): Shape (m, n) - Input features
        y (ndarray): Shape (m,) - Target values
        w (ndarray): Shape (n,) - Weight vector
        b (scalar): Bias parameter
        
    Returns:
        total_cost (scalar): The cost J(w,b)
    """
    m = X.shape[0]
    
    # Vectorized prediction: f_wb for all examples
    f_wb = np.dot(X, w) + b
    
    # Vectorized cost calculation (Mean Squared Error)
    total_cost = np.sum((f_wb - y) ** 2) / (2 * m)
    
    return total_cost


def compute_gradient(X, y, w, b):
    """
    Compute gradients for linear regression using vectorized operations.
    
    dj_dw = (1/m) * Σ(f_w,b(X^(i)) - y^(i)) * X^(i)
    dj_db = (1/m) * Σ(f_w,b(X^(i)) - y^(i))
    
    Parameters:
        X (ndarray): Shape (m, n) - Input features
        y (ndarray): Shape (m,) - Target values
        w (ndarray): Shape (n,) - Weight vector
        b (scalar): Bias parameter
        
    Returns:
        dj_dw (ndarray): Shape (n,) - Gradient with respect to w
        dj_db (scalar): Gradient with respect to b
    """
    m = X.shape[0]
    
    # Vectorized prediction
    f_wb = np.dot(X, w) + b
    
    # Vectorized error calculation
    error = f_wb - y
    
    # Vectorized gradient calculations
    dj_dw = np.dot(X.T, error) / m
    dj_db = np.sum(error) / m
    
    return dj_dw, dj_db


def gradient_descent(X, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
    Perform gradient descent to learn w and b for linear regression.
    
    Updates w and b by taking num_iters gradient steps with learning rate alpha.
    
    Parameters:
        X (ndarray): Shape (m, n) - Input features
        y (ndarray): Shape (m,) - Target values
        w_in (ndarray): Shape (n,) - Initial weight vector
        b_in (scalar): Initial bias parameter
        alpha (scalar): Learning rate
        num_iters (int): Number of iterations to run gradient descent
        cost_function: Function to compute cost
        gradient_function: Function to compute gradients
        
    Returns:
        w (ndarray): Updated weight vector after gradient descent
        b (scalar): Updated bias parameter after gradient descent
        J_history (list): History of cost values
    """
    J_history = []
    w = w_in.copy()
    b = b_in
    
    for i in range(num_iters):
        # Calculate gradients
        dj_dw, dj_db = gradient_function(X, y, w, b)
        
        # Update parameters simultaneously
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        # Save cost J at each iteration
        if i < 100000:
            cost = cost_function(X, y, w, b)
            J_history.append(cost)
        
        # Print cost every 10% of iterations
        if i % (num_iters // 10) == 0 or i == num_iters - 1:
            cost = cost_function(X, y, w, b)
            print(f"Iteration {i:4d}: Cost {cost:8.4f}")
    
    return w, b, J_history


class LinearRegression:
    """
    Linear Regression model with feature scaling using Coursera ML Specialization notation.
    
    Attributes:
        w (ndarray): Weight parameters (shape: n,)
        b (scalar): Bias parameter
        alpha (float): Learning rate
        num_iters (int): Number of iterations for gradient descent
        J_history (list): History of cost values during training
        mu (ndarray): Mean values for feature normalization
        sigma (ndarray): Standard deviation values for feature normalization
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
        self.mu = None
        self.sigma = None
    
    def fit(self, X_train, y_train, w_init=None, b_init=0.0):
        """
        Train the model using gradient descent with feature normalization.
        
        Parameters:
            X_train (ndarray): Shape (m, n) - Training features
            y_train (ndarray): Shape (m,) - Training targets
            w_init (ndarray): Initial weight vector (initialized to zeros if None)
            b_init (scalar): Initial bias parameter (default: 0.0)
            
        Returns:
            self: Returns the instance for method chaining
        """
        # Ensure inputs are numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train).flatten()
        
        m, n = X_train.shape
        
        # Initialize weights to zeros if not provided
        if w_init is None:
            w_init = np.zeros(n)
        else:
            w_init = np.array(w_init)
        
        # Z-Score Normalization
        print(f"\nFeature Normalization:")
        print(f"  Computing mean (μ) and std dev (σ) for each feature...")
        X_norm, self.mu, self.sigma = normalize_features(X_train)
        print(f"  Normalization complete!")
        
        print(f"\nTraining Linear Regression Model with Feature Scaling:")
        print(f"  Number of training examples (m): {m}")
        print(f"  Number of features (n): {n}")
        print(f"  Learning rate (alpha): {self.alpha}")
        print(f"  Number of iterations: {self.num_iters}")
        print(f"\nRunning gradient descent...\n")
        
        # Run gradient descent on normalized features
        self.w, self.b, self.J_history = gradient_descent(
            X_norm, y_train, w_init, b_init,
            self.alpha, self.num_iters,
            compute_cost, compute_gradient
        )
        
        print(f"\nTraining complete!")
        print(f"  Final parameters: w shape = {self.w.shape}, b = {self.b:.6f}")
        print(f"  Final cost: {self.J_history[-1]:.6f}")
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        f_w,b(X) = w·X + b
        
        Parameters:
            X (ndarray): Shape (m, n) or (n,) - Input features
            
        Returns:
            predictions (ndarray): Predicted values
        """
        if self.w is None or self.b is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        X = np.array(X)
        
        # Handle 1D input
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Normalize features using training set statistics
        X_norm = (X - self.mu) / self.sigma
        
        # Compute predictions
        predictions = np.dot(X_norm, self.w) + self.b
        
        return predictions
    
    def score(self, X, y):
        """
        Calculate R² score (coefficient of determination).
        
        R² = 1 - (SS_res / SS_tot)
        
        Where:
        - SS_res = Σ(y_i - ŷ_i)² (residual sum of squares)
        - SS_tot = Σ(y_i - ȳ)² (total sum of squares)
        
        Parameters:
            X (ndarray): Feature values
            y (ndarray): True target values
            
        Returns:
            r2_score (float): R² score (0 to 1, higher is better)
        """
        y = np.array(y).flatten()
        y_pred = self.predict(X)
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        r2_score = 1 - (ss_res / ss_tot)
        
        return r2_score
    
    def mean_squared_error(self, X, y):
        """
        Calculate Mean Squared Error.
        
        MSE = (1/m) * Σ(y_i - ŷ_i)²
        
        Parameters:
            X (ndarray): Feature values
            y (ndarray): True target values
            
        Returns:
            mse (float): Mean squared error
        """
        y = np.array(y).flatten()
        y_pred = self.predict(X)
        m = len(y)
        
        mse = np.sum((y - y_pred) ** 2) / m
        
        return mse
    
    def root_mean_squared_error(self, X, y):
        """
        Calculate Root Mean Squared Error.
        
        RMSE = sqrt(MSE)
        
        Parameters:
            X (ndarray): Feature values
            y (ndarray): True target values
            
        Returns:
            rmse (float): Root mean squared error
        """
        mse = self.mean_squared_error(X, y)
        rmse = np.sqrt(mse)
        
        return rmse
    
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
        plt.title(f'Linear Regression: Cost vs Iterations (alpha={self.alpha})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and test sets.
    
    Parameters:
        X (ndarray): Feature matrix
        y (ndarray): Target vector
        test_size (float): Fraction of data for test set
        random_state (int): Random seed
        
    Returns:
        X_train, X_test, y_train, y_test: Split datasets
    """
    np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def plot_regression_results(X, y, y_pred, feature_idx=0, title="Linear Regression Results"):
    """
    Visualize regression results (for 1D or 2D features).
    
    Parameters:
        X (ndarray): Feature values
        y (ndarray): True target values
        y_pred (ndarray): Predicted values
        feature_idx (int): Feature index for plotting (default: 0)
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Sort by feature for better visualization
    sorted_idx = np.argsort(X[:, feature_idx])
    X_sorted = X[sorted_idx]
    y_sorted = y[sorted_idx]
    y_pred_sorted = y_pred[sorted_idx]
    
    # Plot data points
    plt.scatter(X_sorted[:, feature_idx], y_sorted, color='blue', alpha=0.6, 
               marker='x', s=100, label='Actual')
    
    # Plot predictions
    plt.plot(X_sorted[:, feature_idx], y_pred_sorted, color='red', linewidth=2, 
            label='Predicted')
    
    # Formatting
    plt.xlabel(f'Feature {feature_idx} (Normalized)', fontsize=12)
    plt.ylabel('Target Value', fontsize=12)
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
    print("LINEAR REGRESSION WITH FEATURE SCALING")
    print("Coursera ML Specialization Implementation")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Create synthetic regression dataset
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("🔬 CREATING SYNTHETIC DATASET")
    print("-" * 70)
    
    np.random.seed(42)
    
    # Generate multivariate regression dataset
    m = 200  # Number of examples
    n = 3    # Number of features
    
    # Generate random features
    X = np.random.randn(m, n) * np.array([100, 50, 20])  # Different scales
    
    # Generate target: y = 2*x1 + 3*x2 - 1.5*x3 + noise
    true_w = np.array([2.0, 3.0, -1.5])
    true_b = 10.0
    y = np.dot(X, true_w) + true_b + np.random.randn(m) * 5
    
    print(f"\nDataset Information:")
    print(f"  Number of examples (m): {m}")
    print(f"  Number of features (n): {n}")
    print(f"  True relationship: y = 2*x1 + 3*x2 - 1.5*x3 + 10 + noise")
    print(f"\n  Feature ranges (before normalization):")
    for i in range(n):
        print(f"    Feature {i}: [{X[:, i].min():.2f}, {X[:, i].max():.2f}]")
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n  Training set: {X_train.shape[0]} examples")
    print(f"  Test set: {X_test.shape[0]} examples")
    
    # -------------------------------------------------------------------------
    # Train model
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("🎓 TRAINING LINEAR REGRESSION MODEL")
    print("-" * 70)
    
    model = LinearRegression(alpha=0.01, num_iters=1000)
    model.fit(X_train, y_train)
    
    # -------------------------------------------------------------------------
    # Model performance
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("📊 MODEL PERFORMANCE")
    print("-" * 70)
    
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    train_rmse = model.root_mean_squared_error(X_train, y_train)
    test_rmse = model.root_mean_squared_error(X_test, y_test)
    
    print(f"\n  Training R² Score: {train_r2:.6f}")
    print(f"  Test R² Score:     {test_r2:.6f}")
    print(f"\n  Training RMSE: {train_rmse:.6f}")
    print(f"  Test RMSE:     {test_rmse:.6f}")
    
    # -------------------------------------------------------------------------
    # Compare learned parameters with true parameters
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("📈 PARAMETER COMPARISON")
    print("-" * 70)
    
    params = model.get_parameters()
    print(f"\n  {'Parameter':<15} {'Learned':<15} {'True (scaled)':<15}")
    print(f"  {'-'*15} {'-'*15} {'-'*15}")
    
    # Note: Due to normalization, learned weights won't directly match true weights
    # But the predictions should be very close
    for i in range(n):
        print(f"  w{i}               {params['w'][i]:<15.6f} {true_w[i]:<15.6f}")
    print(f"  b                {params['b']:<15.6f} {true_b:<15.6f}")
    
    # -------------------------------------------------------------------------
    # Sample predictions
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("🎯 SAMPLE PREDICTIONS")
    print("-" * 70)
    
    # Select first 5 test examples
    sample_indices = np.arange(min(5, X_test.shape[0]))
    X_sample = X_test[sample_indices]
    y_sample = y_test[sample_indices]
    
    predictions = model.predict(X_sample)
    errors = y_sample - predictions
    
    print(f"\n  {'Actual':<12} {'Predicted':<12} {'Error':<12}")
    print(f"  {'-'*12} {'-'*12} {'-'*12}")
    for true, pred, err in zip(y_sample, predictions, errors):
        print(f"  {true:<12.4f} {pred:<12.4f} {err:<12.4f}")
    
    # -------------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("📊 GENERATING VISUALIZATIONS")
    print("-" * 70)
    print("\nPlotting regression results and cost history...")
    
    # Get predictions on test set
    y_pred_test = model.predict(X_test)
    
    # Plot regression results (using first feature)
    plot_regression_results(
        X_test, y_test, y_pred_test, feature_idx=0,
        title="Linear Regression with Feature Scaling - Test Set"
    )
    
    # Plot cost history
    model.plot_cost_history()
    
    print("\n" + "=" * 70)
    print("✅ DEMONSTRATION COMPLETE!")
    print("=" * 70)
