"""
Logistic Regression with L2 Regularization
===========================================

This module implements Logistic Regression with L2 Regularization following the 
Coursera Machine Learning Specialization (Andrew Ng) notation to prevent overfitting.

Notation:
---------
    - X_train: Training example matrix (m rows, n columns)
    - y_train: Training example targets (binary: 0 or 1, vector of length m)
    - X[i], y[i]: The i-th training example and target
    - m: Number of training examples
    - n: Number of features in each example
    - w: Parameter: weights (vector of length n)
    - b: Parameter: bias (scalar)
    - alpha: Learning rate
    - lambda (λ): Regularization parameter (controls regularization strength)
    - z: Linear combination z = w·X + b
    - g(z): Sigmoid function g(z) = 1/(1 + e^(-z))
    - f_wb: Model prediction f_w,b(X) = g(w·X + b)
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
    J(w,b) = -(1/m) * Σ[y^(i) * log(f_w,b) + (1 - y^(i)) * log(1 - f_w,b)] + (λ/2m) * Σ w_j²
    
    Where:
    - First term: Binary cross-entropy (logistic loss)
    - Second term: L2 regularization penalty (does NOT include bias b)
    - λ: Regularization parameter

Regularized Gradients:
---------------------
    dj_dw = (1/m) * Σ(f_w,b(X^(i)) - y^(i)) * X^(i) + (λ/m) * w
    dj_db = (1/m) * Σ(f_w,b(X^(i)) - y^(i))  (no regularization term)

Implementation:
---------------
This implementation uses vectorized NumPy operations with L2 regularization
to prevent overfitting in binary classification. Feature scaling is included
for faster convergence.

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
    if mu is None:
        mu = np.mean(X_train, axis=0)
    if sigma is None:
        sigma = np.std(X_train, axis=0)
    
    X_norm = (X_train - mu) / sigma
    
    return X_norm, mu, sigma


def compute_cost(X, y, w, b, lambda_reg=0.0):
    """
    Compute regularized cost function for logistic regression.
    
    J(w,b) = -(1/m) * Σ[y*log(f_wb) + (1-y)*log(1-f_wb)] + (λ/2m) * Σ w_j²
    
    Parameters:
        X (ndarray): Shape (m, n) - Input features
        y (ndarray): Shape (m,) - Target values (0 or 1)
        w (ndarray): Shape (n,) - Weight vector
        b (scalar): Bias parameter
        lambda_reg (float): Regularization parameter λ (default: 0.0)
        
    Returns:
        total_cost (scalar): The regularized cost J(w,b)
    """
    m = X.shape[0]
    
    # Vectorized prediction: f_wb for all examples
    z = np.dot(X, w) + b
    f_wb = sigmoid(z)
    
    # Prevent log(0) by clipping predictions
    eps = 1e-15
    f_wb = np.clip(f_wb, eps, 1 - eps)
    
    # Binary cross-entropy cost (logistic loss)
    bce_cost = -(1 / m) * np.sum(y * np.log(f_wb) + (1 - y) * np.log(1 - f_wb))
    
    # Regularization cost: (λ/2m) * Σ w_j²
    # Note: bias b is NOT regularized
    reg_cost = (lambda_reg / (2 * m)) * np.sum(w ** 2)
    
    # Total cost
    total_cost = bce_cost + reg_cost
    
    return total_cost


def compute_gradient(X, y, w, b, lambda_reg=0.0):
    """
    Compute regularized gradients for logistic regression.
    
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
    z = np.dot(X, w) + b
    f_wb = sigmoid(z)
    
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
    Perform gradient descent with regularization for logistic regression.
    
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


class LogisticRegressionRegularized:
    """
    Logistic Regression model with L2 Regularization.
    
    Uses Coursera ML Specialization notation with Z-Score normalization
    and L2 regularization to prevent overfitting in binary classification.
    
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
        Initialize the Logistic Regression model with regularization.
        
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
            y_train (ndarray): Shape (m,) - Training targets (0 or 1)
            w_init (ndarray): Initial weight vector (initialized to zeros if None)
            b_init (scalar): Initial bias parameter (default: 0.0)
            
        Returns:
            self: Returns the instance for method chaining
        """
        # Ensure inputs are numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train).flatten().astype(int)
        
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
        
        print(f"\nTraining Logistic Regression Model with L2 Regularization:")
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
    
    def predict(self, X, threshold=0.5):
        """
        Make binary predictions using the trained model.
        
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
        
        # Handle 1D input
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Normalize features using training set statistics
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
        Calculate accuracy score on classification data.
        
        Accuracy = (number of correct predictions) / (total predictions)
        
        Parameters:
            X (ndarray): Feature values
            y (ndarray): True target values
            
        Returns:
            accuracy (float): Fraction of correct predictions
        """
        y = np.array(y).flatten().astype(int)
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        
        return accuracy
    
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
        plt.title(f'Logistic Regression with Regularization: Cost vs Iterations\n(alpha={self.alpha}, λ={self.lambda_reg})',
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


def create_synthetic_classification_data(n_samples=200, n_features=2, random_state=42):
    """
    Create a synthetic binary classification dataset.
    
    Parameters:
        n_samples (int): Number of examples
        n_features (int): Number of features
        random_state (int): Random seed
        
    Returns:
        X (ndarray): Shape (n_samples, n_features) - Feature matrix
        y (ndarray): Shape (n_samples,) - Binary labels (0 or 1)
    """
    np.random.seed(random_state)
    
    n_samples_per_class = n_samples // 2
    
    # Class 0: centered at (-1, -1, ...)
    center_0 = np.array([-1.0] * n_features)
    X_class0 = np.random.randn(n_samples_per_class, n_features) + center_0
    y_class0 = np.zeros(n_samples_per_class)
    
    # Class 1: centered at (1, 1, ...)
    center_1 = np.array([1.0] * n_features)
    X_class1 = np.random.randn(n_samples_per_class, n_features) + center_1
    y_class1 = np.ones(n_samples_per_class)
    
    # Combine and shuffle
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([y_class0, y_class1]).astype(int)
    
    # Shuffle indices
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y


# =============================================================================
# EXAMPLE USAGE AND DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("LOGISTIC REGRESSION WITH L2 REGULARIZATION")
    print("Coursera ML Specialization Implementation")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Create synthetic binary classification dataset
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("🔬 CREATING SYNTHETIC DATASET")
    print("-" * 80)
    
    np.random.seed(42)
    
    # Generate dataset with 300 samples, 5 features, 2 classes
    X, y = create_synthetic_classification_data(
        n_samples=300,
        n_features=5,
        random_state=42
    )
    
    print(f"\nDataset Information:")
    print(f"  Number of examples (m): {X.shape[0]}")
    print(f"  Number of features (n): {X.shape[1]}")
    print(f"  Class distribution: Class 0: {np.sum(y == 0)}, Class 1: {np.sum(y == 1)}")
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n  Training set: {X_train.shape[0]} examples ({np.sum(y_train == 1)} positive)")
    print(f"  Test set: {X_test.shape[0]} examples ({np.sum(y_test == 1)} positive)")
    
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
        
        model = LogisticRegressionRegularized(alpha=0.1, num_iters=500, lambda_reg=lam)
        model.fit(X_train, y_train)
        models.append(model)
    
    # -------------------------------------------------------------------------
    # Compare model performance
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("📊 MODEL PERFORMANCE COMPARISON")
    print("-" * 80)
    
    print(f"\n  {'λ':<8} {'Train Acc':<12} {'Test Acc':<12} {'Weight Sum':<15} {'Overfitting':<12}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*15} {'-'*12}")
    
    for lam, model in zip(lambda_values, models):
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        params = model.get_parameters()
        weight_sum = np.sum(np.abs(params['w']))
        overfitting = train_acc - test_acc
        
        print(f"  {lam:<8.2f} {train_acc:<12.6f} {test_acc:<12.6f} {weight_sum:<15.6f} {overfitting:<12.6f}")
    
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
    
    test_acc_scores = [model.score(X_test, y_test) for model in models]
    best_idx = np.argmax(test_acc_scores)
    best_lambda = lambda_values[best_idx]
    best_model = models[best_idx]
    
    print(f"\nBest λ value: {best_lambda}")
    print(f"  Training Accuracy: {best_model.score(X_train, y_train):.6f}")
    print(f"  Test Accuracy: {test_acc_scores[best_idx]:.6f}")
    
    # -------------------------------------------------------------------------
    # Sample predictions from best model
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("🎯 SAMPLE PREDICTIONS (Best Model λ={})".format(best_lambda))
    print("-" * 80)
    
    sample_indices = np.arange(min(10, X_test.shape[0]))
    X_sample = X_test[sample_indices]
    y_sample = y_test[sample_indices]
    
    predictions = best_model.predict(X_sample)
    probabilities = best_model.predict_proba(X_sample)
    
    print(f"\n  {'True':<6} {'Predicted':<12} {'Probability':<15} {'Correct':<8}")
    print(f"  {'-'*6} {'-'*12} {'-'*15} {'-'*8}")
    for true, pred, prob in zip(y_sample, predictions, probabilities):
        prob_val = prob if np.isscalar(prob) else prob[0]
        correct = "✓" if true == pred else "✗"
        print(f"  {true:<6} {pred:<12} {prob_val:<15.4f} {correct:<8}")
    
    # -------------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("📊 GENERATING VISUALIZATIONS")
    print("-" * 80)
    print("\nPlotting model comparisons and cost histories...")
    
    # Plot 1: Cost histories for all models
    plt.figure(figsize=(12, 6))
    for lam, model in zip(lambda_values, models):
        plt.plot(model.J_history, linewidth=2, label=f'λ={lam}')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost J(w,b)', fontsize=12)
    plt.title('Regularization Effect on Training Cost (Logistic Regression)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Train vs Test Accuracy comparison
    train_acc_scores = [model.score(X_train, y_train) for model in models]
    
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(lambda_values))
    width = 0.35
    
    plt.bar(x_pos - width/2, train_acc_scores, width, label='Training Accuracy', alpha=0.8)
    plt.bar(x_pos + width/2, test_acc_scores, width, label='Test Accuracy', alpha=0.8)
    
    plt.xlabel('Regularization Parameter (λ)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Train vs Test Accuracy: Effect of Regularization (Logistic)', fontsize=14)
    plt.xticks(x_pos, [f'{lam}' for lam in lambda_values])
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim([0.85, 1.0])
    plt.tight_layout()
    plt.show()
    
    # Plot 3: Weight magnitudes across λ values
    weight_sums = []
    for model in models:
        params = model.get_parameters()
        weight_sums.append(np.sum(np.abs(params['w'])))
    
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_values, weight_sums, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Regularization Parameter (λ)', fontsize=12)
    plt.ylabel('Sum of Absolute Weight Magnitudes', fontsize=12)
    plt.title('Weight Shrinkage: Effect of Regularization Parameter', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 80)
    print("✅ DEMONSTRATION COMPLETE!")
    print("=" * 80)
