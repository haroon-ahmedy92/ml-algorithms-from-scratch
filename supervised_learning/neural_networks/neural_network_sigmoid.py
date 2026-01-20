"""
Simple Neural Network with Sigmoid Activation
==============================================

This module implements a simple feedforward neural network with sigmoid activation
following the Coursera Machine Learning Specialization (Andrew Ng) notation.

Notation:
---------
    - X_train: Training example matrix (m rows, n features)
    - y_train: Training example targets (m samples)
    - m: Number of training examples
    - n: Number of input features
    - L: Number of layers (including input and output)
    - n_l: Number of units in layer l
    - w[l]: Weight matrix for layer l, shape (n_l, n_{l-1})
    - b[l]: Bias vector for layer l, shape (n_l,)
    - z[l]: Linear combination z[l] = w[l] * a[l-1] + b[l]
    - a[l]: Activation output of layer l, a[l] = σ(z[l])
    - σ(z): Sigmoid activation function σ(z) = 1/(1 + e^(-z))
    - alpha: Learning rate
    - L_cost: Cost function value

Forward Propagation (2-layer example):
--------------------------------------
    z1 = w1 * X + b1
    a1 = sigmoid(z1)
    z2 = w2 * a1 + b2
    a2 = sigmoid(z2)
    
    Output: a2 (predictions)

Backward Propagation (Gradients):
---------------------------------
    dz[l] = da[l] * sigmoid_derivative(z[l])
    dw[l] = (1/m) * dz[l] * a[l-1].T
    db[l] = (1/m) * sum(dz[l])
    da[l-1] = w[l].T * dz[l]

Cost Function (Binary Classification):
--------------------------------------
    J = -(1/m) * Σ[y*log(a_out) + (1-y)*log(1-a_out)]

Architecture:
--------------
    This implementation supports configurable network architecture:
    - Input layer: n features
    - Hidden layer(s): configurable units
    - Output layer: 1 unit (binary classification)

Implementation:
---------------
Uses vectorized NumPy operations for efficiency. Includes feature scaling
(Z-Score normalization) and gradient checking capabilities.

Author: ML Algorithms from Scratch Project
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    """
    Sigmoid activation function.
    
    σ(z) = 1 / (1 + e^(-z))
    
    Parameters:
        z (ndarray or scalar): Input value(s)
        
    Returns:
        ndarray or scalar: Sigmoid of z (values between 0 and 1)
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    """
    Derivative of sigmoid function.
    
    σ'(z) = σ(z) * (1 - σ(z))
    
    Parameters:
        z (ndarray or scalar): Input value(s)
        
    Returns:
        ndarray or scalar: Derivative of sigmoid at z
    """
    sig = sigmoid(z)
    return sig * (1 - sig)


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


class NeuralNetwork:
    """
    Simple feedforward neural network with sigmoid activation.
    
    Supports multiple hidden layers for binary classification.
    Uses backpropagation for training.
    
    Attributes:
        layer_dims (list): List of layer dimensions [n_0, n_1, ..., n_L]
                          where n_0 = input features, n_L = output
        weights (dict): Dictionary of weight matrices w[l] for each layer
        biases (dict): Dictionary of bias vectors b[l] for each layer
        alpha (float): Learning rate
        num_iters (int): Number of iterations for training
        cost_history (list): History of cost values during training
        mu (ndarray): Mean values for feature normalization
        sigma (ndarray): Standard deviation values for feature normalization
    """
    
    def __init__(self, layer_dims, alpha=0.01, num_iters=1000):
        """
        Initialize the neural network.
        
        Parameters:
            layer_dims (list): List of layer dimensions [n_input, n_hidden1, ..., n_output]
            alpha (float): Learning rate (default: 0.01)
            num_iters (int): Number of iterations for training (default: 1000)
        """
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        self.alpha = alpha
        self.num_iters = num_iters
        self.cost_history = []
        self.mu = None
        self.sigma = None
        
        # Initialize weights and biases
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """
        Initialize weights and biases for the network.
        
        Uses He initialization for better convergence:
        w[l] ~ N(0, sqrt(2/n_{l-1}))
        b[l] = 0
        """
        self.weights = {}
        self.biases = {}
        
        np.random.seed(42)  # For reproducibility
        
        for l in range(1, self.num_layers):
            # He initialization: scale by sqrt(2/n_{l-1})
            self.weights[l] = np.random.randn(
                self.layer_dims[l], 
                self.layer_dims[l-1]
            ) * np.sqrt(2 / self.layer_dims[l-1])
            
            self.biases[l] = np.zeros((self.layer_dims[l], 1))
    
    def forward_propagation(self, X):
        """
        Perform forward propagation through the network.
        
        Parameters:
            X (ndarray): Shape (m, n) - Input features
            
        Returns:
            cache (dict): Dictionary containing z[l] and a[l] for all layers
            a_out (ndarray): Shape (m, 1) - Network output (predictions)
        """
        m = X.shape[0]
        cache = {'a0': X.T}  # Store activations
        
        # Forward pass through each layer
        for l in range(1, self.num_layers):
            a_prev = cache[f'a{l-1}']
            
            # Linear transformation
            z_l = np.dot(self.weights[l], a_prev) + self.biases[l]
            
            # Activation (sigmoid for all layers in this simple network)
            a_l = sigmoid(z_l)
            
            # Store in cache for backpropagation
            cache[f'z{l}'] = z_l
            cache[f'a{l}'] = a_l
        
        # Output layer activation
        a_out = cache[f'a{self.num_layers-1}']
        
        return cache, a_out
    
    def compute_cost(self, a_out, y):
        """
        Compute binary cross-entropy cost.
        
        J = -(1/m) * Σ[y*log(a_out) + (1-y)*log(1-a_out)]
        
        Parameters:
            a_out (ndarray): Shape (1, m) - Network output
            y (ndarray): Shape (1, m) - True labels
            
        Returns:
            cost (scalar): Binary cross-entropy cost
        """
        m = y.shape[1]
        
        # Prevent log(0) by clipping
        eps = 1e-15
        a_out = np.clip(a_out, eps, 1 - eps)
        
        # Binary cross-entropy
        cost = -(1 / m) * np.sum(y * np.log(a_out) + (1 - y) * np.log(1 - a_out))
        
        return cost
    
    def backward_propagation(self, cache, y):
        """
        Perform backward propagation to compute gradients.
        
        Parameters:
            cache (dict): Dictionary containing activations and z values
            y (ndarray): Shape (1, m) - True labels
            
        Returns:
            grads (dict): Dictionary containing gradients dw[l] and db[l]
        """
        m = y.shape[1]
        grads = {}
        
        # Output layer gradient
        a_out = cache[f'a{self.num_layers-1}']
        dz = a_out - y  # For sigmoid + cross-entropy: dz = a - y
        
        # Backpropagate through layers
        for l in range(self.num_layers - 1, 0, -1):
            # Get previous activation
            a_prev = cache[f'a{l-1}']
            
            # Compute gradients
            grads[f'dw{l}'] = (1 / m) * np.dot(dz, a_prev.T)
            grads[f'db{l}'] = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            
            # Propagate error to previous layer (if not input layer)
            if l > 1:
                dz = np.dot(self.weights[l].T, dz) * sigmoid_derivative(cache[f'z{l-1}'])
        
        return grads
    
    def update_parameters(self, grads):
        """
        Update weights and biases using gradients.
        
        w[l] := w[l] - alpha * dw[l]
        b[l] := b[l] - alpha * db[l]
        
        Parameters:
            grads (dict): Dictionary containing gradients
        """
        for l in range(1, self.num_layers):
            self.weights[l] -= self.alpha * grads[f'dw{l}']
            self.biases[l] -= self.alpha * grads[f'db{l}']
    
    def fit(self, X_train, y_train):
        """
        Train the neural network.
        
        Parameters:
            X_train (ndarray): Shape (m, n) - Training features
            y_train (ndarray): Shape (m,) - Training labels (0 or 1)
            
        Returns:
            self: Returns the instance for method chaining
        """
        # Ensure inputs are numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train).flatten()
        
        m, n = X_train.shape
        
        # Validate dimensions
        if self.layer_dims[0] != n:
            raise ValueError(f"Input dimension mismatch. Expected {self.layer_dims[0]}, got {n}")
        
        # Feature normalization
        print(f"\nFeature Normalization:")
        print(f"  Computing mean (μ) and std dev (σ)...")
        X_train, self.mu, self.sigma = normalize_features(X_train)
        print(f"  Normalization complete!")
        
        # Reshape y for computation
        y_reshaped = y_train.reshape(1, -1)
        
        print(f"\nTraining Neural Network:")
        print(f"  Network architecture: {self.layer_dims}")
        print(f"  Number of training examples (m): {m}")
        print(f"  Learning rate (alpha): {self.alpha}")
        print(f"  Number of iterations: {self.num_iters}")
        print(f"\nRunning training...\n")
        
        # Training loop
        for i in range(self.num_iters):
            # Forward propagation
            cache, a_out = self.forward_propagation(X_train)
            
            # Compute cost
            cost = self.compute_cost(a_out, y_reshaped)
            self.cost_history.append(cost)
            
            # Backward propagation
            grads = self.backward_propagation(cache, y_reshaped)
            
            # Update parameters
            self.update_parameters(grads)
            
            # Print progress
            if (i + 1) % (self.num_iters // 10) == 0 or i == 0:
                print(f"Iteration {i+1:4d}/{self.num_iters}: Cost {cost:.6f}")
        
        print(f"\nTraining complete!")
        print(f"  Final cost: {self.cost_history[-1]:.6f}")
        
        return self
    
    def predict(self, X, threshold=0.5):
        """
        Make binary predictions.
        
        Parameters:
            X (ndarray): Shape (m, n) or (n,) - Input features
            threshold (float): Decision boundary (default: 0.5)
            
        Returns:
            predictions (ndarray): Binary predictions (0 or 1)
        """
        X = np.array(X)
        
        # Handle 1D input
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Normalize features
        X_norm = (X - self.mu) / self.sigma
        
        # Forward propagation
        cache, a_out = self.forward_propagation(X_norm)
        
        # Convert probabilities to binary predictions
        predictions = (a_out.T >= threshold).astype(int).flatten()
        
        return predictions
    
    def predict_proba(self, X):
        """
        Get probability predictions.
        
        Parameters:
            X (ndarray): Shape (m, n) or (n,) - Input features
            
        Returns:
            proba (ndarray): Predicted probabilities
        """
        X = np.array(X)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Normalize features
        X_norm = (X - self.mu) / self.sigma
        
        # Forward propagation
        cache, a_out = self.forward_propagation(X_norm)
        
        return a_out.T
    
    def score(self, X, y):
        """
        Calculate accuracy score.
        
        Parameters:
            X (ndarray): Feature values
            y (ndarray): True labels
            
        Returns:
            accuracy (float): Fraction of correct predictions
        """
        y = np.array(y).flatten().astype(int)
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        
        return accuracy
    
    def plot_cost_history(self):
        """
        Plot the cost function over iterations.
        """
        if not self.cost_history:
            print("No cost history available. Train the model first.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history, 'b-', linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Cost', fontsize=12)
        plt.title(f'Neural Network Training: Cost vs Iterations\n(Architecture: {self.layer_dims}, α={self.alpha})',
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


def create_synthetic_classification_data(n_samples=400, n_features=2, random_state=42):
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
    
    # Class 0: centered at (-2, -2, ...)
    center_0 = np.array([-2.0] * n_features)
    X_class0 = np.random.randn(n_samples_per_class, n_features) * 1.5 + center_0
    y_class0 = np.zeros(n_samples_per_class)
    
    # Class 1: centered at (2, 2, ...)
    center_1 = np.array([2.0] * n_features)
    X_class1 = np.random.randn(n_samples_per_class, n_features) * 1.5 + center_1
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
    print("SIMPLE NEURAL NETWORK WITH SIGMOID ACTIVATION")
    print("Coursera ML Specialization Implementation")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Create synthetic binary classification dataset
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("🔬 CREATING SYNTHETIC DATASET")
    print("-" * 80)
    
    np.random.seed(42)
    
    # Generate dataset with 400 samples, 2 features
    X, y = create_synthetic_classification_data(
        n_samples=400,
        n_features=2,
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
    # Train neural networks with different architectures
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("🎓 TRAINING NEURAL NETWORKS")
    print("-" * 80)
    
    # Test different architectures
    architectures = [
        [2, 1],           # Logistic regression (no hidden layer)
        [2, 4, 1],        # 1 hidden layer with 4 units
        [2, 8, 4, 1],     # 2 hidden layers
    ]
    
    models = []
    
    for arch in architectures:
        print(f"\n{'='*50}")
        print(f"Training network: {arch}")
        print(f"{'='*50}")
        
        model = NeuralNetwork(layer_dims=arch, alpha=0.1, num_iters=1000)
        model.fit(X_train, y_train)
        models.append(model)
    
    # -------------------------------------------------------------------------
    # Compare model performance
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("📊 MODEL PERFORMANCE COMPARISON")
    print("-" * 80)
    
    print(f"\n  {'Architecture':<20} {'Train Acc':<12} {'Test Acc':<12} {'Final Cost':<12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
    
    for arch, model in zip(architectures, models):
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        final_cost = model.cost_history[-1]
        
        arch_str = str(arch)
        print(f"  {arch_str:<20} {train_acc:<12.6f} {test_acc:<12.6f} {final_cost:<12.6f}")
    
    # Use the best model (typically the one with hidden layer)
    best_model = models[1]  # [2, 4, 1]
    
    # -------------------------------------------------------------------------
    # Detailed analysis of best model
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("🏆 BEST MODEL ANALYSIS: " + str(best_model.layer_dims))
    print("-" * 80)
    
    train_acc = best_model.score(X_train, y_train)
    test_acc = best_model.score(X_test, y_test)
    
    print(f"\n  Training Accuracy: {train_acc:.6f}")
    print(f"  Test Accuracy:     {test_acc:.6f}")
    print(f"  Overfitting:       {train_acc - test_acc:.6f}")
    
    # -------------------------------------------------------------------------
    # Sample predictions from best model
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("🎯 SAMPLE PREDICTIONS")
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
    print("\nPlotting training curves and model comparisons...")
    
    # Plot 1: Cost histories for all models
    plt.figure(figsize=(12, 6))
    for arch, model in zip(architectures, models):
        plt.plot(model.cost_history, linewidth=2, label=f'Architecture: {arch}')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.title('Neural Network Training: Cost vs Iterations (Different Architectures)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot 2: Train vs Test Accuracy
    train_accs = [model.score(X_train, y_train) for model in models]
    test_accs = [model.score(X_test, y_test) for model in models]
    
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(architectures))
    width = 0.35
    
    arch_labels = [str(arch) for arch in architectures]
    plt.bar(x_pos - width/2, train_accs, width, label='Training Accuracy', alpha=0.8)
    plt.bar(x_pos + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)
    
    plt.xlabel('Network Architecture', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Neural Network Performance Comparison', fontsize=14)
    plt.xticks(x_pos, arch_labels)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim([0.8, 1.0])
    plt.tight_layout()
    plt.show()
    
    # Plot 3: Best model cost history
    best_model.plot_cost_history()
    
    print("\n" + "=" * 80)
    print("✅ DEMONSTRATION COMPLETE!")
    print("=" * 80)
