"""
Logistic Regression from Scratch
=================================

This module implements Logistic Regression following the Coursera Machine Learning 
Specialization (Andrew Ng) notation and structure.

Notation:
---------
    - x_train: Input features (training data)
    - y_train: Target values (binary: 0 or 1)
    - m: Number of training examples
    - n: Number of features
    - w: Weight parameters (vector)
    - b: Bias parameter (scalar)
    - alpha: Learning rate
    - z: Linear combination z = w·x + b
    - g(z): Sigmoid function g(z) = 1/(1 + e^(-z))
    - f_wb: Prediction f_w,b(x) = g(w·x + b)
    - dj_dw: Gradient of cost function with respect to w
    - dj_db: Gradient of cost function with respect to b

Cost Function:
--------------
    J(w,b) = -(1/m) * Σ[y^(i) * log(f_w,b(x^(i))) + (1 - y^(i)) * log(1 - f_w,b(x^(i)))]
    
where f_w,b(x^(i)) = sigmoid(w·x^(i) + b)

Implementation:
---------------
This implementation uses vectorized NumPy operations for efficient computation.
Supports both univariate and multivariate logistic regression.

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
        ndarray or scalar: Sigmoid of z, same shape as z
    """
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, w, b):
    """
    Compute cost function for logistic regression using vectorized operations.
    
    J(w,b) = -(1/m) * Σ[y^(i) * log(f_w,b(x^(i))) + (1 - y^(i)) * log(1 - f_w,b(x^(i)))]
    
    Parameters:
        X (ndarray): Shape (m, n) - Input features
        y (ndarray): Shape (m,) - Target values (binary: 0 or 1)
        w (ndarray): Shape (n,) - Weight parameters
        b (scalar): Bias parameter
        
    Returns:
        total_cost (scalar): The cost J(w,b)
    """
    m = X.shape[0]
    
    # Vectorized prediction: f_wb for all examples at once
    z = np.dot(X, w) + b
    f_wb = sigmoid(z)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-8
    
    # Vectorized cost calculation (binary cross-entropy)
    cost = -np.sum(y * np.log(f_wb + epsilon) + (1 - y) * np.log(1 - f_wb + epsilon)) / m
    
    return cost


def compute_gradient(X, y, w, b):
    """
    Compute gradient for logistic regression using vectorized operations.
    
    ∂J/∂w_j = (1/m) * Σ(f_w,b(x^(i)) - y^(i)) * x_j^(i)
    ∂J/∂b = (1/m) * Σ(f_w,b(x^(i)) - y^(i))
    
    Parameters:
        X (ndarray): Shape (m, n) - Input features
        y (ndarray): Shape (m,) - Target values
        w (ndarray): Shape (n,) - Weight parameters
        b (scalar): Bias parameter
        
    Returns:
        dj_dw (ndarray): Shape (n,) - Gradient of cost with respect to w
        dj_db (scalar): Gradient of cost with respect to b
    """
    m, n = X.shape
    
    # Vectorized prediction for all examples
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
    Perform gradient descent to learn w and b.
    
    Updates w and b by taking num_iters gradient steps with learning rate alpha.
    
    Parameters:
        X (ndarray): Shape (m, n) - Input features
        y (ndarray): Shape (m,) - Target values
        w_in (ndarray): Shape (n,) - Initial weight parameters
        b_in (scalar): Initial bias parameter
        alpha (scalar): Learning rate
        num_iters (int): Number of iterations to run gradient descent
        cost_function: Function to compute cost
        gradient_function: Function to compute gradients
        
    Returns:
        w (ndarray): Shape (n,) - Updated weight parameters after gradient descent
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
        if i < 100000:  # Prevent resource exhaustion
            cost = cost_function(X, y, w, b)
            J_history.append(cost)
        
        # Print cost every 10% of iterations
        if i % max(1, num_iters // 10) == 0 or i == num_iters - 1:
            cost = cost_function(X, y, w, b)
            print(f"Iteration {i:4d}: Cost {cost:8.6f}")
    
    return w, b, J_history


class LogisticRegression:
    """
    Logistic Regression model following Coursera ML Specialization notation.
    
    Attributes:
        w (ndarray): Weight parameters
        b (scalar): Bias parameter
        alpha (float): Learning rate
        num_iters (int): Number of iterations for gradient descent
        J_history (list): History of cost values during training
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
        self.n_features = None
    
    def fit(self, X_train, y_train, w_init=None, b_init=0.0):
        """
        Train the model using gradient descent.
        
        Parameters:
            X_train (ndarray): Shape (m, n) or (m,) - Training features
            y_train (ndarray): Shape (m,) - Training targets (binary: 0 or 1)
            w_init (ndarray): Shape (n,) - Initial weight parameters (default: zeros)
            b_init (scalar): Initial bias parameter (default: 0.0)
            
        Returns:
            self: Returns the instance for method chaining
        """
        # Ensure inputs are numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train).flatten()
        
        # Handle 1D input (single feature)
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        
        m, n = X_train.shape
        self.n_features = n
        
        # Initialize weights if not provided
        if w_init is None:
            w_init = np.zeros(n)
        else:
            w_init = np.array(w_init).flatten()
        
        print(f"\nTraining Logistic Regression Model:")
        print(f"  Number of training examples (m): {m}")
        print(f"  Number of features (n): {n}")
        print(f"  Learning rate (alpha): {self.alpha}")
        print(f"  Number of iterations: {self.num_iters}")
        print(f"\nRunning gradient descent...\n")
        
        # Run gradient descent
        self.w, self.b, self.J_history = gradient_descent(
            X_train, y_train, w_init, b_init, 
            self.alpha, self.num_iters, 
            compute_cost, compute_gradient
        )
        
        print(f"\nTraining complete!")
        print(f"  Final parameters:")
        print(f"    w = {self.w}")
        print(f"    b = {self.b:.4f}")
        print(f"  Final cost: {self.J_history[-1]:.6f}")
        
        return self
    
    def predict_proba(self, X):
        """
        Predict probabilities using the trained model.
        
        f_w,b(x) = sigmoid(w·x + b)
        
        Parameters:
            X (ndarray): Shape (m, n) or (m,) - Input features
            
        Returns:
            ndarray: Shape (m,) - Predicted probabilities
        """
        if self.w is None or self.b is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        X = np.array(X)
        
        # Handle 1D input
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        z = np.dot(X, self.w) + self.b
        return sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels using the trained model.
        
        Parameters:
            X (ndarray): Shape (m, n) or (m,) - Input features
            threshold (float): Decision threshold (default: 0.5)
            
        Returns:
            ndarray: Shape (m,) - Predicted class labels (0 or 1)
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)
    
    def compute_cost(self, X, y):
        """
        Compute the cost for the current parameters.
        
        Parameters:
            X (ndarray): Shape (m, n) - Input features
            y (ndarray): Shape (m,) - Target values
            
        Returns:
            scalar: The cost J(w,b)
        """
        if self.w is None or self.b is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return compute_cost(X, y, self.w, self.b)
    
    def score(self, X, y):
        """
        Calculate accuracy score.
        
        Accuracy = (Number of correct predictions) / (Total predictions)
        
        Parameters:
            X (ndarray): Feature values
            y (ndarray): True target values
            
        Returns:
            float: Accuracy score (0 to 1)
        """
        y_pred = self.predict(X)
        y = np.array(y).flatten()
        
        return np.mean(y_pred == y)
    
    def get_parameters(self):
        """
        Get the model parameters.
        
        Returns:
            dict: Dictionary with 'w' and 'b' keys
        """
        if self.w is None or self.b is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        return {
            'w': self.w.copy(),
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


def plot_decision_boundary(X, y, model, title="Logistic Regression Decision Boundary"):
    """
    Visualize the decision boundary for 2D data.
    
    Parameters:
        X (ndarray): Shape (m, 2) - Feature values
        y (ndarray): Shape (m,) - Target values
        model (LogisticRegression): Trained model
        title (str): Plot title
    """
    if X.shape[1] != 2:
        print("Decision boundary plot requires exactly 2 features.")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Plot training data
    class_0 = y == 0
    class_1 = y == 1
    
    plt.scatter(X[class_0, 0], X[class_0, 1], color='red', alpha=0.6, 
                marker='o', s=100, label='Class 0', edgecolors='k')
    plt.scatter(X[class_1, 0], X[class_1, 1], color='blue', alpha=0.6, 
                marker='x', s=100, label='Class 1', linewidths=2)
    
    # Plot decision boundary
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    h = 0.01  # Step size in mesh
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                           np.arange(x2_min, x2_max, h))
    
    Z = model.predict_proba(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    
    # Plot contours
    plt.contourf(xx1, xx2, Z, levels=[0, 0.5, 1], alpha=0.2, colors=['red', 'blue'])
    plt.contour(xx1, xx2, Z, levels=[0.5], colors='black', linewidths=2)
    
    # Formatting
    plt.xlabel('Feature 1 (x₁)', fontsize=12)
    plt.ylabel('Feature 2 (x₂)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_sigmoid():
    """
    Plot the sigmoid activation function.
    """
    z = np.linspace(-10, 10, 200)
    g_z = sigmoid(z)
    
    plt.figure(figsize=(10, 6))
    plt.plot(z, g_z, 'b-', linewidth=2)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Decision threshold')
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.xlabel('z', fontsize=12)
    plt.ylabel('g(z) = 1/(1 + e⁻ᶻ)', fontsize=12)
    plt.title('Sigmoid Activation Function', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.show()


# =============================================================================
# EXAMPLE USAGE AND DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("LOGISTIC REGRESSION - COURSERA ML SPECIALIZATION IMPLEMENTATION")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Example 1: Simple Binary Classification (2 features)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXAMPLE 1: BINARY CLASSIFICATION WITH 2 FEATURES")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate synthetic binary classification dataset
    m = 100
    
    # Class 0: centered around (2, 2)
    X_class0 = np.random.randn(m // 2, 2) * 0.8 + [2, 2]
    y_class0 = np.zeros(m // 2)
    
    # Class 1: centered around (5, 5)
    X_class1 = np.random.randn(m // 2, 2) * 0.8 + [5, 5]
    y_class1 = np.ones(m // 2)
    
    # Combine
    X_train = np.vstack([X_class0, X_class1])
    y_train = np.hstack([y_class0, y_class1])
    
    # Shuffle
    indices = np.random.permutation(m)
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    print(f"\n📊 Dataset Information:")
    print(f"   Number of training examples (m): {m}")
    print(f"   Number of features (n): {X_train.shape[1]}")
    print(f"   Class distribution: {np.sum(y_train == 0)} examples in class 0, "
          f"{np.sum(y_train == 1)} examples in class 1")
    
    # -------------------------------------------------------------------------
    # Test sigmoid function
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("🔬 TESTING SIGMOID FUNCTION")
    print("-" * 70)
    
    test_values = np.array([-10, -1, 0, 1, 10])
    sigmoid_values = sigmoid(test_values)
    
    print(f"\n   {'z':<10} {'g(z) = sigmoid(z)':<20}")
    print(f"   {'-'*10} {'-'*20}")
    for z_val, g_val in zip(test_values, sigmoid_values):
        print(f"   {z_val:<10} {g_val:<20.6f}")
    
    # -------------------------------------------------------------------------
    # Test cost and gradient functions
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("🔬 TESTING COST AND GRADIENT FUNCTIONS")
    print("-" * 70)
    
    # Initialize parameters
    n = X_train.shape[1]
    w_test = np.zeros(n)
    b_test = 0.0
    
    # Test compute_cost
    initial_cost = compute_cost(X_train, y_train, w_test, b_test)
    print(f"\nInitial cost with w={w_test}, b={b_test}: {initial_cost:.6f}")
    
    # Test compute_gradient
    dj_dw, dj_db = compute_gradient(X_train, y_train, w_test, b_test)
    print(f"Initial gradients:")
    print(f"  dj_dw = {dj_dw}")
    print(f"  dj_db = {dj_db:.6f}")
    
    # -------------------------------------------------------------------------
    # Train model using gradient descent
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("🎓 TRAINING MODEL WITH GRADIENT DESCENT")
    print("-" * 70)
    
    model = LogisticRegression(alpha=0.1, num_iters=1000)
    model.fit(X_train, y_train)
    
    # Get learned parameters
    params = model.get_parameters()
    accuracy = model.score(X_train, y_train)
    
    print(f"\n📈 Model Performance:")
    print(f"   Training Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # -------------------------------------------------------------------------
    # Make predictions
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("🎯 SAMPLE PREDICTIONS")
    print("-" * 70)
    
    X_test = np.array([[2, 2], [5, 5], [3.5, 3.5]])
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    print(f"\n   {'X₁':<10} {'X₂':<10} {'Probability':<15} {'Predicted Class':<15}")
    print(f"   {'-'*10} {'-'*10} {'-'*15} {'-'*15}")
    for x_val, prob, pred in zip(X_test, y_proba, y_pred):
        print(f"   {x_val[0]:<10.2f} {x_val[1]:<10.2f} {prob:<15.6f} {pred:<15}")
    
    # -------------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("📊 GENERATING VISUALIZATIONS")
    print("-" * 70)
    print("\nPlotting decision boundary, cost history, and sigmoid function...")
    
    # Plot 1: Sigmoid function
    plot_sigmoid()
    
    # Plot 2: Decision boundary
    plot_decision_boundary(X_train, y_train, model, 
                          title="Logistic Regression - Decision Boundary")
    
    # Plot 3: Cost history
    model.plot_cost_history()
    
    # -------------------------------------------------------------------------
    # Example 2: Single Feature Classification
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXAMPLE 2: BINARY CLASSIFICATION WITH 1 FEATURE")
    print("=" * 70)
    
    np.random.seed(42)
    
    # Generate 1D dataset
    m_1d = 60
    X_1d_class0 = np.random.randn(m_1d // 2) * 0.5 + 2
    X_1d_class1 = np.random.randn(m_1d // 2) * 0.5 + 5
    
    X_train_1d = np.hstack([X_1d_class0, X_1d_class1])
    y_train_1d = np.hstack([np.zeros(m_1d // 2), np.ones(m_1d // 2)])
    
    # Shuffle
    indices_1d = np.random.permutation(m_1d)
    X_train_1d = X_train_1d[indices_1d]
    y_train_1d = y_train_1d[indices_1d]
    
    print(f"\n📊 Dataset Information:")
    print(f"   Number of training examples (m): {m_1d}")
    print(f"   Number of features (n): 1")
    
    # Train model
    print("\n" + "-" * 70)
    print("🎓 TRAINING MODEL")
    print("-" * 70)
    
    model_1d = LogisticRegression(alpha=0.1, num_iters=1000)
    model_1d.fit(X_train_1d, y_train_1d)
    
    accuracy_1d = model_1d.score(X_train_1d, y_train_1d)
    print(f"\n📈 Training Accuracy: {accuracy_1d:.4f} ({accuracy_1d*100:.2f}%)")
    
    # Visualize 1D classification
    print("\n" + "-" * 70)
    print("📊 VISUALIZING 1D CLASSIFICATION")
    print("-" * 70)
    
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    class_0_1d = y_train_1d == 0
    class_1_1d = y_train_1d == 1
    
    plt.scatter(X_train_1d[class_0_1d], np.zeros(np.sum(class_0_1d)), 
                color='red', alpha=0.6, marker='o', s=100, label='Class 0', edgecolors='k')
    plt.scatter(X_train_1d[class_1_1d], np.zeros(np.sum(class_1_1d)), 
                color='blue', alpha=0.6, marker='x', s=100, label='Class 1', linewidths=2)
    
    # Plot decision boundary
    x_range = np.linspace(X_train_1d.min() - 1, X_train_1d.max() + 1, 300)
    probas = model_1d.predict_proba(x_range)
    
    plt.plot(x_range, probas, 'g-', linewidth=2, label='P(y=1|x)')
    plt.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Decision threshold')
    
    plt.xlabel('Feature x', fontsize=12)
    plt.ylabel('Probability / Class', fontsize=12)
    plt.title('Logistic Regression - 1D Classification', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 70)
    print("✅ DEMONSTRATION COMPLETE!")
    print("=" * 70)
