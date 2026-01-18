"""
Linear Regression with L2 Regularization (Ridge Regression)
===========================================================

This module implements Linear Regression with L2 Regularization following the 
Coursera Machine Learning Specialization (Andrew Ng) notation to prevent overfitting.

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
    - lambda (λ): Regularization parameter (controls regularization strength)
    - f_wb: Model prediction f_w,b(X) = w·X + b
    - J(w,b): Cost function with regularization
    - dj_dw: Gradient of cost function with respect to w
    - dj_db: Gradient of cost function with respect to b

Z-Score Normalization (Feature Scaling):
-----------------------------------------
    To speed up gradient descent, features are scaled so they have a mean (μ) of 0
    and a standard deviation (σ) of 1.
    
    x_norm^(i)_j = (x^(i)_j - μ_j) / σ_j

Regularized Cost Function (L2 Regularization):
-----------------------------------------------
    J(w,b) = (1/2m) * Σ(f_w,b(X^(i)) - y^(i))² + (λ/2m) * Σ w_j²
    
    Where:
    - First term: Standard Mean Squared Error
    - Second term: L2 regularization penalty (does NOT include bias b)
    - λ: Regularization parameter (higher λ = more regularization)

Regularized Gradients:
---------------------
    dj_dw = (1/m) * Σ(f_w,b(X^(i)) - y^(i)) * X^(i) + (λ/m) * w
    dj_db = (1/m) * Σ(f_w,b(X^(i)) - y^(i))  (no regularization term)

Implementation:
---------------
This implementation uses vectorized NumPy operations with L2 regularization
to prevent overfitting on training data. Feature scaling is included for
faster convergence.

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
    if mu is None:
        mu = np.mean(X_train, axis=0)
    if sigma is None:
        sigma = np.std(X_train, axis=0)
    
    X_norm = (X_train - mu) / sigma
    
    return X_norm, mu, sigma


def compute_cost(X, y, w, b, lambda_reg=0.0):
    """
    Compute cost function with L2 regularization.
    
    J(w,b) = (1/2m) * Σ(f_w,b(X^(i)) - y^(i))² + (λ/2m) * Σ w_j²
    
    Parameters:
        X (ndarray): Shape (m, n) - Input features
        y (ndarray): Shape (m,) - Target values
        w (ndarray): Shape (n,) - Weight vector
        b (scalar): Bias parameter
        lambda_reg (float): Regularization parameter λ (default: 0.0)
        
    Returns:
        total_cost (scalar): The regularized cost J(w,b)
    """
    m = X.shape[0]
    
    # Vectorized prediction: f_wb for all examples
    f_wb = np.dot(X, w) + b
    
    # Cost from prediction error (MSE)
    mse_cost = np.sum((f_wb - y) ** 2) / (2 * m)
    
    # Regularization cost: (λ/2m) * Σ w_j²
    # Note: bias b is NOT regularized
    reg_cost = (lambda_reg / (2 * m)) * np.sum(w ** 2)
    
    # Total cost
    total_cost = mse_cost + reg_cost
    
    return total_cost


def compute_gradient(X, y, w, b, lambda_reg=0.0):
    """
    Compute gradients with L2 regularization.
    
    dj_dw = (1/m) * Σ(f_w,b(X^(i)) - y^(i)) * X^(i) + (λ/m) * w
    dj_db = (1/m) * Σ(f_w,b(X^(i)) - y^(i))
    
    Parameters:
        X (ndarray): Shape (m, n) - Input features
        y (ndarray): Shape (m,) - Target values
        w (ndarray): Shape (n,) - Weight vector
        b (scalar): Bias parameter
        lambda_reg (float): Regularization parameter λ (default: 0.0)
        
    Returns:
        dj_dw (ndarray): Shape (n,) - Gradient with respect to w
        dj_db (scalar): Gradient with respect to b
    """
    m = X.shape[0]
    
    # Vectorized prediction
    f_wb = np.dot(X, w) + b
    
    # Vectorized error calculation
    error = f_wb - y
    
    # Gradient for w: includes regularization term
    dj_dw = np.dot(X.T, error) / m + (lambda_reg / m) * w
    
    # Gradient for b: no regularization term
    dj_db = np.sum(error) / m
    
    return dj_dw, dj_db


def gradient_descent(X, y, w_in, b_in, alpha, num_iters, lambda_reg, 
                     cost_function, gradient_function):
    """
    Perform gradient descent with regularization.
    
    Updates w and b by taking num_iters gradient steps with learning rate alpha.
    
    Parameters:
        X (ndarray): Shape (m, n) - Input features
        y (ndarray): Shape (m,) - Target values
        w_in (ndarray): Shape (n,) - Initial weight vector
        b_in (scalar): Initial bias parameter
        alpha (scalar): Learning rate
        num_iters (int): Number of iterations to run gradient descent
        lambda_reg (float): Regularization parameter λ
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
        # Calculate gradients with regularization
        dj_dw, dj_db = gradient_function(X, y, w, b, lambda_reg)
        
        # Update parameters simultaneously
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        # Save cost J at each iteration
        if i < 100000:
            cost = cost_function(X, y, w, b, lambda_reg)
            J_history.append(cost)
        
        # Print cost every 10% of iterations
        if i % (num_iters // 10) == 0 or i == num_iters - 1:
            cost = cost_function(X, y, w, b, lambda_reg)
            print(f"Iteration {i:4d}: Cost {cost:8.4f}")
    
    return w, b, J_history


class LinearRegressionRegularized:
    """
    Linear Regression model with L2 Regularization (Ridge Regression).
    
    Uses Coursera ML Specialization notation with Z-Score normalization
    and L2 regularization to prevent overfitting.
    
    Attributes:
        w (ndarray): Weight parameters (shape: n,)
        b (scalar): Bias parameter
        alpha (float): Learning rate
        num_iters (int): Number of iterations for gradient descent
        lambda_reg (float): Regularization parameter
        J_history (list): History of cost values during training
        mu (ndarray): Mean values for feature normalization
        sigma (ndarray): Standard deviation values for feature normalization
    """
    
    def __init__(self, alpha=0.01, num_iters=1000, lambda_reg=0.0):
        """
        Initialize the Linear Regression model with regularization.
        
        Parameters:
            alpha (float): Learning rate (default: 0.01)
            num_iters (int): Number of iterations for gradient descent (default: 1000)
            lambda_reg (float): Regularization parameter λ (default: 0.0 = no regularization)
        """
        self.w = None
        self.b = None
        self.alpha = alpha
        self.num_iters = num_iters
        self.lambda_reg = lambda_reg
        self.J_history = []
        self.mu = None
        self.sigma = None
    
    def fit(self, X_train, y_train, w_init=None, b_init=0.0):
        """
        Train the model using gradient descent with regularization.
        
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
        
        print(f"\nTraining Linear Regression Model with L2 Regularization:")
        print(f"  Number of training examples (m): {m}")
        print(f"  Number of features (n): {n}")
        print(f"  Learning rate (alpha): {self.alpha}")
        print(f"  Regularization parameter (lambda): {self.lambda_reg}")
        print(f"  Number of iterations: {self.num_iters}")
        print(f"\nRunning gradient descent...\n")
        
        # Run gradient descent on normalized features with regularization
        self.w, self.b, self.J_history = gradient_descent(
            X_norm, y_train, w_init, b_init,
            self.alpha, self.num_iters, self.lambda_reg,
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
        
        Parameters:
            X (ndarray): Feature values
            y (ndarray): True target values
            
        Returns:
            r2_score (float): R² score
        """
        y = np.array(y).flatten()
        y_pred = self.predict(X)
        
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        r2_score = 1 - (ss_res / ss_tot)
        
        return r2_score
    
    def mean_squared_error(self, X, y):
        """
        Calculate Mean Squared Error (without regularization term).
        
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
            dict: Dictionary with 'w', 'b', and 'lambda' keys
        """
        if self.w is None or self.b is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        return {
            'w': self.w,
            'b': self.b,
            'lambda': self.lambda_reg
        }
    
    def plot_cost_history(self):
        """
        Plot the regularized cost function J(w,b) over iterations.
        """
        if not self.J_history:
            print("No cost history available. Train the model first.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.J_history, 'b-', linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Cost J(w,b)', fontsize=12)
        plt.title(f'Linear Regression with Regularization: Cost vs Iterations\n(alpha={self.alpha}, λ={self.lambda_reg})', 
                 fontsize=14)
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


def plot_regularization_comparison(X_test, y_test, models, model_names, title="Regularization Comparison"):
    """
    Compare predictions from models with different regularization parameters.
    
    Parameters:
        X_test (ndarray): Test features
        y_test (ndarray): Test targets
        models (list): List of trained models
        model_names (list): Names of models for legend
        title (str): Plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Sort by first feature for visualization
    sorted_idx = np.argsort(X_test[:, 0])
    X_sorted = X_test[sorted_idx]
    y_sorted = y_test[sorted_idx]
    
    # Plot actual data
    plt.scatter(X_sorted[:, 0], y_sorted, color='black', alpha=0.5, 
               marker='x', s=100, label='Actual', zorder=5)
    
    # Plot predictions from each model
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    for model, name, color in zip(models, model_names, colors):
        y_pred = model.predict(X_sorted)
        plt.plot(X_sorted[:, 0], y_pred, color=color, linewidth=2, label=name)
    
    # Formatting
    plt.xlabel('Feature (Normalized)', fontsize=12)
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
    print("=" * 80)
    print("LINEAR REGRESSION WITH L2 REGULARIZATION (RIDGE REGRESSION)")
    print("Coursera ML Specialization Implementation")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Create synthetic regression dataset with polynomial features
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("🔬 CREATING SYNTHETIC DATASET")
    print("-" * 80)
    
    np.random.seed(42)
    
    # Generate data that will benefit from regularization
    m = 100  # Number of examples
    X = np.random.randn(m, 5) * 10  # 5 features with different scales
    
    # Generate target with polynomial relationship to show overfitting potential
    true_w = np.array([2.0, 1.5, 0.5, 0.2, 0.1])  # Small coefficients
    true_b = 5.0
    y = np.dot(X, true_w) + true_b + np.random.randn(m) * 2
    
    print(f"\nDataset Information:")
    print(f"  Number of examples (m): {m}")
    print(f"  Number of features (n): {X.shape[1]}")
    print(f"  True coefficients decay: [2.0, 1.5, 0.5, 0.2, 0.1]")
    print(f"  Note: Smaller coefficients tend to overfit")
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\n  Training set: {X_train.shape[0]} examples")
    print(f"  Test set: {X_test.shape[0]} examples")
    
    # -------------------------------------------------------------------------
    # Train models with different regularization parameters
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("🎓 TRAINING MODELS WITH DIFFERENT REGULARIZATION PARAMETERS")
    print("-" * 80)
    
    # Different regularization strengths
    lambda_values = [0.0, 0.01, 0.1, 1.0]
    models = []
    
    for lam in lambda_values:
        print(f"\n{'='*40}")
        print(f"Training with λ = {lam}")
        print(f"{'='*40}")
        
        model = LinearRegressionRegularized(alpha=0.1, num_iters=500, lambda_reg=lam)
        model.fit(X_train, y_train)
        models.append(model)
    
    # -------------------------------------------------------------------------
    # Compare model performance
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("📊 MODEL PERFORMANCE COMPARISON")
    print("-" * 80)
    
    print(f"\n  {'λ':<8} {'Train R²':<12} {'Test R²':<12} {'Train RMSE':<12} {'Test RMSE':<12} {'Overfitting':<12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    for lam, model in zip(lambda_values, models):
        train_r2 = model.score(X_train, y_train)
        test_r2 = model.score(X_test, y_test)
        train_rmse = model.root_mean_squared_error(X_train, y_train)
        test_rmse = model.root_mean_squared_error(X_test, y_test)
        overfitting = train_r2 - test_r2  # Positive = overfitting
        
        print(f"  {lam:<8.2f} {train_r2:<12.6f} {test_r2:<12.6f} {train_rmse:<12.6f} {test_rmse:<12.6f} {overfitting:<12.6f}")
    
    # -------------------------------------------------------------------------
    # Analyze weight magnitudes
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("📈 WEIGHT MAGNITUDE ANALYSIS")
    print("-" * 80)
    print("\nHow regularization shrinks weights (prevents overfitting):\n")
    
    for lam, model in zip(lambda_values, models):
        params = model.get_parameters()
        w_norms = np.abs(params['w'])
        w_sum = np.sum(w_norms)
        
        w_str = ", ".join([f"{w:.4f}" for w in w_norms])
        print(f"λ = {lam:<6.2f} | Weight magnitudes: [{w_str}] | Sum: {w_sum:.4f}")
    
    # -------------------------------------------------------------------------
    # Best model selection
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("🏆 BEST MODEL SELECTION")
    print("-" * 80)
    
    test_r2_scores = [model.score(X_test, y_test) for model in models]
    best_idx = np.argmax(test_r2_scores)
    best_lambda = lambda_values[best_idx]
    best_model = models[best_idx]
    
    print(f"\nBest λ value: {best_lambda}")
    print(f"  Test R² Score: {test_r2_scores[best_idx]:.6f}")
    print(f"  Test RMSE: {best_model.root_mean_squared_error(X_test, y_test):.6f}")
    
    # -------------------------------------------------------------------------
    # Sample predictions from best model
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("🎯 SAMPLE PREDICTIONS (Best Model λ={})".format(best_lambda))
    print("-" * 80)
    
    sample_indices = np.arange(min(5, X_test.shape[0]))
    X_sample = X_test[sample_indices]
    y_sample = y_test[sample_indices]
    
    predictions = best_model.predict(X_sample)
    errors = y_sample - predictions
    
    print(f"\n  {'Actual':<12} {'Predicted':<12} {'Error':<12}")
    print(f"  {'-'*12} {'-'*12} {'-'*12}")
    for true, pred, err in zip(y_sample, predictions, errors):
        print(f"  {true:<12.4f} {pred:<12.4f} {err:<12.4f}")
    
    # -------------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("📊 GENERATING VISUALIZATIONS")
    print("-" * 80)
    print("\nPlotting model comparisons and cost histories...")
    
    # Plot 1: Predictions comparison
    plot_regularization_comparison(
        X_test, y_test, models,
        [f'λ={lam}' for lam in lambda_values],
        title="Effect of Regularization on Predictions"
    )
    
    # Plot 2: Cost histories for all models
    plt.figure(figsize=(12, 6))
    for lam, model in zip(lambda_values, models):
        plt.plot(model.J_history, linewidth=2, label=f'λ={lam}')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost J(w,b)', fontsize=12)
    plt.title('Regularization Effect on Training Cost', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot 3: Train vs Test R² comparison
    plt.figure(figsize=(10, 6))
    train_r2_scores = [model.score(X_train, y_train) for model in models]
    
    x_pos = np.arange(len(lambda_values))
    width = 0.35
    
    plt.bar(x_pos - width/2, train_r2_scores, width, label='Training R²', alpha=0.8)
    plt.bar(x_pos + width/2, test_r2_scores, width, label='Test R²', alpha=0.8)
    
    plt.xlabel('Regularization Parameter (λ)', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title('Train vs Test R² Score: Effect of Regularization', fontsize=14)
    plt.xticks(x_pos, [f'{lam}' for lam in lambda_values])
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 80)
    print("✅ DEMONSTRATION COMPLETE!")
    print("=" * 80)
