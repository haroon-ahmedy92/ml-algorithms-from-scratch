"""
Multi-class Neural Network Classification with Logits
======================================================

This script demonstrates multi-class classification using TensorFlow with the numerically
stable "logits" approach. Instead of using softmax activation in the output layer,
we use LINEAR activation and set from_logits=True in the loss function.

Key Concepts:
-------------
1. **Multi-class Classification**: Assigning an input to one of K classes (K > 2).
2. **Logits**: Raw, unnormalized predictions before softmax conversion.
3. **Numerical Stability**: Using from_logits=True prevents numerical overflow/underflow.
4. **Feature Engineering via Hidden Layers**: Layer 1 creates new features that Layer 2 uses.

Mathematical Foundation:
------------------------
For multi-class classification, the softmax function converts logits to probabilities:
$$P(y=k | \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

Where:
- $z_k$ is the logit (raw score) for class k
- The denominator normalizes across all K classes

Numerical Stability Issue:
--------------------------
Computing softmax directly can cause overflow when $z_k$ is large:
$$e^{1000} \approx \infty \quad \text{(overflow)}$$

Solution: The "from_logits=True" approach uses the Log-Sum-Exp trick:
$$\log \left( \frac{e^{z_k}}{\sum_j e^{z_j}} \right) = z_k - \log \left( \sum_j e^{z_j} \right)$$

This avoids computing large exponentials explicitly.

Author: ML Algorithms from Scratch Project
Date: January 26, 2026
License: MIT
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import os

# =============================================================================
# SECTION 1: DATA PREPARATION
# =============================================================================

def create_and_save_dataset():
    """
    Creates a 4-class dataset using sklearn's make_blobs and saves to disk.
    
    Dataset Specification:
    ----------------------
    - n_samples: 100 total examples
    - centers: 4 cluster centers at specific (x, y) coordinates
    - cluster_std: 1.0 (controls spread around each center)
    
    The dataset is saved as:
    - X_train.npy: Shape (100, 2) - Feature matrix
    - y_train.npy: Shape (100,) - Labels [0, 1, 2, 3]
    """
    print("=" * 70)
    print("Multi-class Classification with Logits (4 Classes)")
    print("=" * 70)
    
    # Create data directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)
    
    # Define cluster centers for 4 classes
    centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
    
    # Generate the dataset
    X_train, y_train = make_blobs(
        n_samples=100,
        centers=centers,
        cluster_std=1.0,
        random_state=42
    )
    
    # Save to disk
    np.save('./data/X_train.npy', X_train)
    np.save('./data/y_train.npy', y_train)
    
    print("\n✓ Dataset created and saved:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  Classes: {np.unique(y_train)}")
    print(f"  Cluster centers: {centers}")
    
    return X_train, y_train

def load_dataset():
    """
    Loads the dataset from .npy files.
    
    This maintains the local-loading workflow pattern used in other scripts.
    
    Returns:
        X_train (ndarray): Shape (100, 2) - Input features
        y_train (ndarray): Shape (100,) - Class labels
    """
    print("\nLoading dataset from files...")
    X_train = np.load('./data/X_train.npy')
    y_train = np.load('./data/y_train.npy')
    
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  Class distribution: {np.bincount(y_train)}")
    
    return X_train, y_train

# =============================================================================
# SECTION 2: MODEL ARCHITECTURE
# =============================================================================

def build_model():
    """
    Builds a 2-layer neural network for multi-class classification.
    
    Architecture:
    -------------
    Input (2 features) -> Layer 1 (2 units, ReLU) -> Layer 2 (4 units, LINEAR)
    
    Layer 1 Role - Feature Engineering:
    ------------------------------------
    The first layer with ReLU activation creates "inflection points" or new features.
    Each neuron in this layer learns to identify regions in the input space.
    
    Mathematical representation:
    $$\mathbf{a}^{[1]} = \text{ReLU}(W^{[1]} \mathbf{x} + b^{[1]})$$
    
    Where:
    - $W^{[1]}$ is (2, 2): maps 2 input features to 2 hidden units
    - ReLU introduces non-linearity: $\text{ReLU}(z) = \max(0, z)$
    - This creates "bending" in the decision boundary
    
    Layer 2 Role - Class Selection:
    --------------------------------
    The output layer takes the 2 new features from Layer 1 and produces 4 logits.
    Each logit represents the "score" for a particular class.
    
    Mathematical representation:
    $$\mathbf{z}^{[2]} = W^{[2]} \mathbf{a}^{[1]} + b^{[2]}$$
    
    Where:
    - $W^{[2]}$ is (2, 4): maps 2 hidden features to 4 class scores
    - $\mathbf{z}^{[2]}$ is the logit vector (unnormalized scores)
    
    Why LINEAR activation in the last layer?
    -----------------------------------------
    Using a LINEAR activation (no activation) in the output layer combined with
    from_logits=True in the loss function is MORE NUMERICALLY STABLE than:
    
    1. Applying softmax in the output layer, then
    2. Using categorical crossentropy loss
    
    Reason: The combined softmax + cross-entropy computation can be simplified
    mathematically to avoid computing large exponentials that cause overflow:
    
    Unstable: $y_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$, then $-\log(y_k)$
    Stable:   Compute $-\log(y_k)$ directly using log-sum-exp trick
    
    TensorFlow handles this automatically when from_logits=True.
    
    Returns:
        model (tf.keras.Model): Compiled Sequential model
    """
    print("\n" + "=" * 70)
    print("Building TensorFlow Model")
    print("=" * 70)
    
    model = Sequential([
        Dense(2, activation='relu', name='L1'),     # Hidden layer: 2 units with ReLU
        Dense(4, activation='linear', name='L2')    # Output layer: 4 units with LINEAR
    ], name='multiclass_model')
    
    # Compile with SparseCategoricalCrossentropy
    # from_logits=True means the model outputs logits, not probabilities
    model.compile(
        loss=SparseCategoricalCrossentropy(from_logits=True),
        optimizer=Adam(learning_rate=0.01),
        metrics=['accuracy']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    return model

# =============================================================================
# SECTION 3: TRAINING
# =============================================================================

def train_model(model, X_train, y_train, epochs=200):
    """
    Trains the model on the dataset.
    
    Parameters:
        model: Compiled Keras model
        X_train: Training features
        y_train: Training labels
        epochs: Number of training iterations
        
    Returns:
        history: Training history object
    """
    print("\n" + "=" * 70)
    print(f"Training for {epochs} epochs...")
    print("=" * 70)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        verbose=1  # Show training progress
    )
    
    final_loss = history.history['loss'][-1]
    final_accuracy = history.history['accuracy'][-1]
    
    print(f"\n✓ Training Complete!")
    print(f"  Final Loss: {final_loss:.4f}")
    print(f"  Final Accuracy: {final_accuracy*100:.2f}%")
    
    return history

# =============================================================================
# SECTION 4: WEIGHT EXTRACTION & ANALYSIS
# =============================================================================

def extract_weights(model):
    """
    Extracts weights and biases from each layer for analysis.
    
    Returns:
        Dictionary containing W1, b1, W2, b2
    """
    print("\n" + "=" * 70)
    print("Extracting Weights from Trained Model")
    print("=" * 70)
    
    # Layer 1 (L1): Hidden layer
    W1, b1 = model.get_layer('L1').get_weights()
    print(f"\nLayer 1 (L1) - Feature Engineering Layer:")
    print(f"  W1 shape: {W1.shape} (2 inputs -> 2 hidden units)")
    print(f"  b1 shape: {b1.shape}")
    print(f"\n  W1 = \n{W1}")
    print(f"\n  b1 = {b1}")
    
    # Layer 2 (L2): Output layer
    W2, b2 = model.get_layer('L2').get_weights()
    print(f"\nLayer 2 (L2) - Class Selection Layer:")
    print(f"  W2 shape: {W2.shape} (2 hidden units -> 4 classes)")
    print(f"  b2 shape: {b2.shape}")
    print(f"\n  W2 = \n{W2}")
    print(f"\n  b2 = {b2}")
    
    print("\n" + "-" * 70)
    print("Interpretation:")
    print("-" * 70)
    print("• Layer 1 (W1, b1):")
    print("    - Creates 2 new 'engineered features' from the original 2 inputs")
    print("    - ReLU activation creates 'inflection points' (non-linear bends)")
    print("    - Each hidden unit learns to activate for specific input regions")
    print("\n• Layer 2 (W2, b2):")
    print("    - Takes the 2 engineered features and produces 4 logits (scores)")
    print("    - Each column of W2 represents how each hidden feature contributes")
    print("      to the score for a particular class")
    print("    - The class with the highest logit is the predicted class")
    
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

# =============================================================================
# SECTION 5: VISUALIZATION
# =============================================================================

def plot_decision_boundaries(model, X_train, y_train):
    """
    Visualizes the decision boundaries learned by the model.
    
    Creates a mesh grid over the input space and colors each region
    according to the predicted class.
    
    Decision Boundary Mathematics:
    ------------------------------
    For each point (x1, x2) in the grid:
    1. Compute hidden activations: $a = \text{ReLU}(W_1 [x_1, x_2]^T + b_1)$
    2. Compute logits: $z = W_2 a + b_2$
    3. Predict class: $\hat{y} = \arg\max(z)$
    
    The boundary occurs where two classes have equal logits.
    """
    print("\n" + "=" * 70)
    print("Generating Decision Boundary Visualization")
    print("=" * 70)
    
    # Create mesh grid
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    
    h = 0.02  # Step size
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict for each point in the mesh
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Get logits from the model
    logits = model.predict(grid_points, verbose=0)
    
    # Convert logits to class predictions
    Z = np.argmax(logits, axis=1)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Plot decision boundaries
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis', levels=3)
    
    # Plot training points
    scatter = plt.scatter(X_train[:, 0], X_train[:, 1], 
                         c=y_train, cmap='viridis', 
                         edgecolors='black', s=100, linewidth=1.5)
    
    plt.colorbar(scatter, label='Class')
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.title('Multi-class Classification Decision Boundaries\n(4 Classes with Logits)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multiclass_decision_boundaries.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved visualization to 'multiclass_decision_boundaries.png'")
    
    # Also display the plot
    plt.show()

def plot_training_history(history):
    """
    Plots the training loss and accuracy over epochs.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(history.history['loss'], linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss over Epochs', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history.history['accuracy'], linewidth=2, color='green')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training Accuracy over Epochs', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved training history to 'training_history.png'")
    
    # Also display the plot
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution pipeline.
    """
    # Step 1: Create and save dataset
    X_train, y_train = create_and_save_dataset()
    
    # Step 2: Load dataset (to maintain local-loading workflow)
    X_train, y_train = load_dataset()
    
    # Step 3: Build model
    model = build_model()
    
    # Step 4: Train model
    history = train_model(model, X_train, y_train, epochs=200)
    
    # Step 5: Extract and analyze weights
    weights = extract_weights(model)
    
    # Step 6: Visualize results
    plot_decision_boundaries(model, X_train, y_train)
    plot_training_history(history)
    
    print("\n" + "=" * 70)
    print("✓ Multi-class Classification Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("1. Layer 1 creates 2 new features via ReLU activation")
    print("2. Layer 2 maps these features to 4 class logits")
    print("3. Using from_logits=True ensures numerical stability")
    print("4. The model learns non-linear decision boundaries for 4 classes")
    print("=" * 70)

if __name__ == "__main__":
    main()
