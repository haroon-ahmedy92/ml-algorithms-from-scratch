"""
Logistic Regression with Feature Scaling (Z-Score Normalization)
=================================================================

This module implements Logistic Regression following the Coursera Machine Learning 
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
    - z: Linear combination z = w·X + b
    - g(z): Sigmoid function g(z) = 1/(1 + e^(-z))
    - f_wb: Model prediction f_w,b(X) = g(w·X + b)
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
    J(w,b) = -(1/m) * Σ[y^(i) * log(f_w,b(X^(i))) + (1 - y^(i)) * log(1 - f_w,b(X^(i)))]
    
where f_w,b(X^(i)) = sigmoid(w·X^(i) + b)

Implementation:
---------------
This implementation uses vectorized NumPy operations for efficient computation.
Supports both univariate and multivariate logistic regression with feature scaling.

Author: ML Algorithms from Scratch Project
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    """
    Compute the sigmoid function.
    
    g(z) = 1/(1 + e^(-z))
    
    Parameters:
        z (ndarray or scalar): Input value(s)
        
    Returns:
        ndarray or scalar: Sigmoid of z (values between 0 and 1)
    """
    return 1 / (1 + np.exp(-z))


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
    Compute cost function for logistic regression using vectorized operations.
    
    J(w,b) = -(1/m) * Σ[y^(i) * log(f_w,b(X^(i))) + (1 - y^(i)) * log(1 - f_w,b(X^(i)))]
    
    Parameters:
        X (ndarray): Shape (m, n) - Input features
        y (ndarray): Shape (m,) - Target values (0 or 1)
        w (ndarray): Shape (n,) - Weight vector
        b (scalar): Bias parameter
        
    Returns:
        total_cost (scalar): The cost J(w,b)
    """
    m = X.shape[0]
    
    # Vectorized prediction: f_wb for all examples
    z = np.dot(X, w) + b
    f_wb = sigmoid(z)
    
    # Vectorized cost calculation (binary cross-entropy)
    # Add small epsilon to prevent log(0)
    eps = 1e-15
    f_wb = np.clip(f_wb, eps, 1 - eps)
    
    total_cost = -(1 / m) * np.sum(y * np.log(f_wb) + (1 - y) * np.log(1 - f_wb))
    
    return total_cost


def compute_gradient(X, y, w, b):
    """
    Compute gradients for logistic regression using vectorized operations.
    
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
    z = np.dot(X, w) + b
    f_wb = sigmoid(z)
    
    # Vectorized error calculation
    error = f_wb - y
    
    # Vectorized gradient calculations
    dj_dw = np.dot(X.T, error) / m
    dj_db = np.sum(error) / m
    
    return dj_dw, dj_db


def gradient_descent(X, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
    Perform gradient descent to learn w and b for logistic regression.
    
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


class LogisticRegression:
    """
    Logistic Regression model with feature scaling using Coursera ML Specialization notation.
    
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
        Initialize the Logistic Regression model.
        
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
            y_train (ndarray): Shape (m,) - Training targets (0 or 1)
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
        
        print(f"\nTraining Logistic Regression Model with Feature Scaling:")
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
    
    def predict(self, X, threshold=0.5):
        """
        Make predictions using the trained model.
        
        f_w,b(X) = sigmoid(w·X + b)
        
        Prediction = 1 if f_w,b >= threshold, else 0
        
        Parameters:
            X (ndarray): Shape (m, n) or (n,) - Input features
            threshold (float): Decision boundary threshold (default: 0.5)
            
        Returns:
            predictions (ndarray): Binary predictions (0 or 1)
        """
        if self.w is None or self.b is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        X = np.array(X)
        
        # Normalize features using training set statistics
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X_norm = (X - self.mu) / self.sigma
        
        # Compute sigmoid
        z = np.dot(X_norm, self.w) + self.b
        f_wb = sigmoid(z)
        
        # Convert probabilities to binary predictions
        predictions = (f_wb >= threshold).astype(int)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Get probability predictions for positive class.
        
        Parameters:
            X (ndarray): Shape (m, n) or (n,) - Input features
            
        Returns:
            proba (ndarray): Predicted probabilities for positive class
        """
        if self.w is None or self.b is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        X = np.array(X)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Normalize features
        X_norm = (X - self.mu) / self.sigma
        
        # Compute sigmoid
        z = np.dot(X_norm, self.w) + self.b
        f_wb = sigmoid(z)
        
        return f_wb
    
    def score(self, X, y):
        """
        Calculate accuracy score on test data.
        
        Parameters:
            X (ndarray): Feature values
            y (ndarray): True target values
            
        Returns:
            accuracy (float): Fraction of correct predictions
        """
        y = np.array(y).flatten()
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        
        return accuracy
    
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
        plt.title(f'Logistic Regression: Cost vs Iterations (alpha={self.alpha})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def plot_decision_boundary(X, y, model, title="Logistic Regression - Decision Boundary"):
    """
    Visualize the decision boundary for 2D classification data.
    
    Parameters:
        X (ndarray): Shape (m, 2) - Feature values (2D only)
        y (ndarray): Target values
        model (LogisticRegression): Trained model
        title (str): Plot title
    """
    if X.shape[1] != 2:
        print("Decision boundary visualization only supported for 2D features.")
        return
    
    # Create mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # Get predictions for mesh points
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points).reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.contour(xx, yy, Z, colors='black', linewidths=0.5)
    
    # Plot data points
    positive = y == 1
    plt.scatter(X[positive, 0], X[positive, 1], color='blue', marker='o', 
               s=100, label='Class 1', alpha=0.7)
    plt.scatter(X[~positive, 0], X[~positive, 1], color='red', marker='x', 
               s=100, label='Class 0', alpha=0.7)
    
    # Formatting
    plt.xlabel('Feature 1 (Normalized)', fontsize=12)
    plt.ylabel('Feature 2 (Normalized)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# EXAMPLE USAGE AND DEMONSTRATION
# =============================================================================

# =============================================================================
# EXAMPLE USAGE AND DEMONSTRATION
# =============================================================================

def create_synthetic_classification_data(n_samples=200, n_features=2, random_state=42):
    """
    Create a synthetic binary classification dataset.
    
    Parameters:
        n_samples (int): Number of examples
        n_features (int): Number of features
        random_state (int): Random seed for reproducibility
        
    Returns:
        X (ndarray): Shape (n_samples, n_features) - Feature matrix
        y (ndarray): Shape (n_samples,) - Binary labels (0 or 1)
    """
    np.random.seed(random_state)
    
    # Generate two clusters of points
    n_samples_per_class = n_samples // 2
    
    # Class 0: centered at (-1, -1)
    X_class0 = np.random.randn(n_samples_per_class, n_features) + np.array([-1, -1])
    y_class0 = np.zeros(n_samples_per_class)
    
    # Class 1: centered at (1, 1)
    X_class1 = np.random.randn(n_samples_per_class, n_features) + np.array([1, 1])
    y_class1 = np.ones(n_samples_per_class)
    
    # Combine and shuffle
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([y_class0, y_class1]).astype(int)
    
    # Shuffle indices
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y


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


if __name__ == "__main__":
    print("=" * 70)
    print("LOGISTIC REGRESSION WITH FEATURE SCALING")
    print("Coursera ML Specialization Implementation")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Create synthetic classification dataset
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("🔬 CREATING SYNTHETIC DATASET")
    print("-" * 70)
    
    np.random.seed(42)
    
    # Generate dataset with 200 samples, 2 features, 2 classes
    X, y = create_synthetic_classification_data(
        n_samples=200,
        n_features=2,
        random_state=42
    )
    
    print(f"\nDataset Information:")
    print(f"  Number of examples (m): {X.shape[0]}")
    print(f"  Number of features (n): {X.shape[1]}")
    print(f"  Class distribution: {np.bincount(y)}")
    print(f"  Feature ranges (before normalization):")
    for i in range(X.shape[1]):
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
    print("🎓 TRAINING LOGISTIC REGRESSION MODEL")
    print("-" * 70)
    
    model = LogisticRegression(alpha=0.1, num_iters=1000)
    model.fit(X_train, y_train)
    
    # -------------------------------------------------------------------------
    # Model performance
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("📊 MODEL PERFORMANCE")
    print("-" * 70)
    
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    print(f"\n  Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # -------------------------------------------------------------------------
    # Sample predictions with probabilities
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("🎯 SAMPLE PREDICTIONS")
    print("-" * 70)
    
    # Select first 5 test examples
    sample_indices = np.arange(min(5, X_test.shape[0]))
    X_sample = X_test[sample_indices]
    y_sample = y_test[sample_indices]
    
    predictions = model.predict(X_sample)
    probabilities = model.predict_proba(X_sample)
    
    print(f"\n  {'True':<6} {'Predicted':<12} {'Probability':<15}")
    print(f"  {'-'*6} {'-'*12} {'-'*15}")
    for true, pred, prob in zip(y_sample, predictions, probabilities):
        pred_val = pred if np.isscalar(pred) else pred[0]
        prob_val = prob if np.isscalar(prob) else prob[0]
        print(f"  {true:<6} {pred_val:<12} {prob_val:<15.4f}")
    
    # -------------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("📈 GENERATING VISUALIZATIONS")
    print("-" * 70)
    print("\nPlotting decision boundary and cost history...")
    
    # Plot decision boundary
    plot_decision_boundary(
        X_test, y_test, model,
        title="Logistic Regression with Z-Score Normalization - Decision Boundary"
    )
    
    # Plot cost history
    model.plot_cost_history()
    
    print("\n" + "=" * 70)
    print("✅ DEMONSTRATION COMPLETE!")
    print("=" * 70)
