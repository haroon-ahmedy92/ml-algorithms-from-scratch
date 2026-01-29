"""
MNIST Multiclass Classification - Handwritten Digit Recognition (0-9)
=======================================================================

This script implements a multiclass neural network for recognizing handwritten digits
using the MNIST-subset dataset. It demonstrates the "Preferred" approach with numerical
stability using from_logits=True.

Educational Objective:
----------------------
Understanding multiclass classification with 10 classes (digits 0-9) using:
1. The numerically stable "logits" approach
2. Custom softmax implementation in NumPy
3. Visualization of predictions on real handwritten digits

Dataset:
--------
- X: (5000, 400) - Each row is a 20x20 grayscale image flattened into 400 pixels
- y: (5000, 1) - Integer labels from 0-9 representing digits

Mathematical Foundation:
------------------------
For multiclass classification with K=10 classes:

Softmax Function:
$$a_j = \frac{e^{z_j}}{\sum_{k=1}^{10} e^{z_k}}$$

Where:
- $z_j$ are the logits (raw scores) from the output layer
- $a_j$ are the probabilities for each class j ∈ {0,1,2,...,9}
- The predicted class is: $\hat{y} = \arg\max_j a_j$

Numerical Stability:
--------------------
Using from_logits=True combines softmax + cross-entropy into a single operation
using the log-sum-exp trick, avoiding overflow/underflow issues.

Author: ML Algorithms from Scratch Project
Date: January 29, 2026
License: MIT
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# =============================================================================
# SECTION 1: DATA LOADING & PREPROCESSING
# =============================================================================

def load_mnist_data():
    """
    Loads the MNIST-subset dataset from local .npy files.
    
    Dataset Structure:
    ------------------
    - X: (5000, 400) array
      * 5000 handwritten digit images
      * Each image: 20×20 pixels = 400 features
      * Pixel values: normalized grayscale [0, 1]
      
    - y: (5000, 1) array
      * Integer labels: 0, 1, 2, ..., 9
      * Each label corresponds to a handwritten digit
      
    Image Unrolling:
    ----------------
    A 20×20 image is stored as a 400-dimensional vector:
    $$\text{Image}_{20 \times 20} \rightarrow \text{Vector}_{400 \times 1}$$
    
    To reconstruct: Reshape vector back to (20, 20) matrix.
    
    Returns:
        X (ndarray): Shape (5000, 400) - Flattened images
        y (ndarray): Shape (5000,) - Integer labels
    """
    print("=" * 70)
    print("MNIST Multiclass Classification - Handwritten Digits (0-9)")
    print("=" * 70)
    
    # Load data from .npy files
    print("\nLoading MNIST-subset dataset...")
    X = np.load('./data/X.npy')
    y = np.load('./data/y.npy')
    
    print(f"\n✓ Dataset Loaded:")
    print(f"  X shape: {X.shape} (5000 images, each 20×20=400 pixels)")
    print(f"  y shape: {y.shape} (Integer labels: 0-9)")
    print(f"  Pixel range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"  Unique labels: {np.unique(y.flatten())}")
    print(f"  Class distribution: {np.bincount(y.flatten())}")
    
    # Flatten y for easier handling (from (5000,1) to (5000,))
    y = y.flatten()
    
    print("\n" + "-" * 70)
    print("Understanding the Data:")
    print("-" * 70)
    print("• Each row in X represents one handwritten digit")
    print("• 400 pixels = 20×20 grayscale image")
    print("• Pixel values are normalized (likely [0,1] or [-0.5,0.5])")
    print("• Labels are integers 0-9 (no one-hot encoding needed)")
    print("• This is a 10-class classification problem")
    print("-" * 70)
    
    return X, y

# =============================================================================
# SECTION 2: CUSTOM SOFTMAX IMPLEMENTATION
# =============================================================================

def my_softmax(z):
    """
    Custom NumPy implementation of the Softmax function.
    
    Mathematical Formula:
    ---------------------
    For input vector z = [z₁, z₂, ..., z₁₀]:
    
    $$a_j = \frac{e^{z_j}}{\sum_{k=1}^{10} e^{z_k}}$$
    
    Numerical Stability Trick:
    --------------------------
    To prevent overflow when z_j are large, subtract the maximum:
    
    $$a_j = \frac{e^{z_j - \max(z)}}{\sum_k e^{z_k - \max(z)}}$$
    
    This doesn't change the result but keeps exponentials reasonable.
    
    Parameters:
        z (ndarray): Input logits, shape (10,) or (batch_size, 10)
        
    Returns:
        a (ndarray): Softmax probabilities, same shape as z
    """
    # Numerical stability: subtract max before exponentiating
    z_shifted = z - np.max(z, axis=-1, keepdims=True)
    
    # Compute exponentials
    exp_z = np.exp(z_shifted)
    
    # Normalize to get probabilities
    a = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    
    return a

def test_softmax_implementations():
    """
    Compares custom softmax with TensorFlow's implementation.
    """
    print("\n" + "=" * 70)
    print("Testing Softmax Implementations")
    print("=" * 70)
    
    # Test vector
    test_logits = np.array([1.0, 2.0, 3.0, 4.0])
    print(f"\nTest logits: {test_logits}")
    
    # Custom implementation
    custom_probs = my_softmax(test_logits)
    print(f"\nCustom softmax:     {custom_probs}")
    
    # TensorFlow implementation
    tf_probs = tf.nn.softmax(test_logits).numpy()
    print(f"TensorFlow softmax: {tf_probs}")
    
    # Check if they match
    max_diff = np.max(np.abs(custom_probs - tf_probs))
    print(f"\nMaximum difference: {max_diff:.2e}")
    
    if max_diff < 1e-6:
        print("✓ Implementations match! (within numerical precision)")
    else:
        print("✗ Implementations differ significantly")
    
    # Verify properties
    print(f"\nVerification:")
    print(f"  Sum of probabilities: {np.sum(custom_probs):.6f} (should be 1.0)")
    print(f"  All values ≥ 0: {np.all(custom_probs >= 0)}")
    print(f"  All values ≤ 1: {np.all(custom_probs <= 1)}")
    print(f"  Highest probability at index: {np.argmax(custom_probs)} (should be 3)")

# =============================================================================
# SECTION 3: MODEL ARCHITECTURE
# =============================================================================

def build_model():
    """
    Builds the multiclass neural network using the preferred approach.
    
    Architecture:
    -------------
    Input (400 pixels) → Layer 1 (25 units, ReLU) → Layer 2 (15 units, ReLU) → Output (10 units, LINEAR)
    
    Why This Architecture?
    ----------------------
    - **Input Layer**: 400 features (20×20 pixels)
    - **Hidden Layers**: ReLU activations create non-linear decision boundaries
    - **Output Layer**: 10 units (one per digit class) with LINEAR activation
    - **No softmax in output**: We use from_logits=True for numerical stability
    
    Mathematical Flow:
    ------------------
    1. Hidden Layer 1: $\mathbf{a}^{[1]} = \text{ReLU}(W^{[1]} \mathbf{x} + b^{[1]})$
    2. Hidden Layer 2: $\mathbf{a}^{[2]} = \text{ReLU}(W^{[2]} \mathbf{a}^{[1]} + b^{[2]})$
    3. Output Layer: $\mathbf{z}^{[3]} = W^{[3]} \mathbf{a}^{[2]} + b^{[3]}$ (logits)
    4. Softmax: $\mathbf{a}^{[3]} = \text{softmax}(\mathbf{z}^{[3]})$ (probabilities)
    5. Prediction: $\hat{y} = \arg\max(\mathbf{a}^{[3]})$
    
    Returns:
        model: Compiled Keras Sequential model
    """
    print("\n" + "=" * 70)
    print("Building Multiclass Neural Network")
    print("=" * 70)
    
    model = Sequential([
        Dense(25, activation='relu', name='L1'),     # Hidden layer 1: 25 units
        Dense(15, activation='relu', name='L2'),     # Hidden layer 2: 15 units
        Dense(10, activation='linear', name='L3_output')  # Output: 10 units (linear)
    ], name='mnist_multiclass_model')
    
    # Compile with the preferred approach
    model.compile(
        loss=SparseCategoricalCrossentropy(from_logits=True),  # Expects logits
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    print("\n" + "-" * 70)
    print("Why This Design?")
    print("-" * 70)
    print("• Input: 400 pixels (20×20 image)")
    print("• Hidden layers: ReLU creates non-linear features")
    print("• Output: 10 logits (one per digit class)")
    print("• Linear activation + from_logits=True: Numerically stable")
    print("• SparseCategoricalCrossentropy: Integer labels (efficient)")
    print("-" * 70)
    
    return model

# =============================================================================
# SECTION 4: TRAINING
# =============================================================================

def train_model(model, X, y, epochs=40):
    """
    Trains the model on the MNIST dataset.
    
    Parameters:
        model: Compiled Keras model
        X, y: Training data
        epochs: Number of training iterations
        
    Returns:
        history: Training history
    """
    print("\n" + "=" * 70)
    print(f"Training Model for {epochs} Epochs")
    print("=" * 70)
    
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,  # 20% for validation
        verbose=1
    )
    
    final_loss = history.history['loss'][-1]
    final_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    
    print(f"\n✓ Training Complete!")
    print(f"  Final Loss: {final_loss:.4f}")
    print(f"  Final Accuracy: {final_accuracy*100:.2f}%")
    print(f"  Validation Accuracy: {final_val_accuracy*100:.2f}%")
    
    return history

# =============================================================================
# SECTION 5: PREDICTION DEMONSTRATION
# =============================================================================

def demonstrate_prediction(model, X, y):
    """
    Demonstrates prediction for a single example (X[1015]).
    
    Shows the complete pipeline:
    1. Raw logits from model
    2. Conversion to probabilities using tf.nn.softmax
    3. Final prediction using argmax
    """
    print("\n" + "=" * 70)
    print("Prediction Demonstration - Single Image")
    print("=" * 70)
    
    # Select example at index 1015
    idx = 1015
    x_sample = X[idx:idx+1]  # Shape (1, 400)
    y_true = y[idx]
    
    print(f"\nAnalyzing image at index {idx}:")
    print(f"  True label: {y_true}")
    
    # Step 1: Get raw logits
    logits = model.predict(x_sample, verbose=0)[0]  # Shape (10,)
    print(f"\nStep 1 - Raw Logits (from output layer):")
    print(f"  {logits}")
    print(f"  Shape: {logits.shape}")
    
    # Step 2: Convert to probabilities
    probabilities = tf.nn.softmax(logits).numpy()
    print(f"\nStep 2 - Probabilities (after softmax):")
    print(f"  {probabilities}")
    print(f"  Sum: {np.sum(probabilities):.6f} (should be 1.0)")
    
    # Step 3: Get predicted class
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]
    
    print(f"\nStep 3 - Final Prediction:")
    print(f"  Predicted digit: {predicted_class}")
    print(f"  Confidence: {confidence*100:.2f}%")
    print(f"  Correct: {'✓' if predicted_class == y_true else '✗'}")
    
    # Show top 3 predictions
    top_indices = np.argsort(probabilities)[::-1][:3]
    print(f"\nTop 3 Predictions:")
    for i, class_idx in enumerate(top_indices):
        prob = probabilities[class_idx]
        print(f"  {i+1}. Digit {class_idx}: {prob*100:.2f}%")
    
    print("\n" + "-" * 70)
    print("Key Insights:")
    print("-" * 70)
    print("• Logits are raw scores (can be any real numbers)")
    print("• Softmax converts logits to valid probabilities")
    print("• argmax finds the class with highest probability")
    print("• For classification, you only need argmax - softmax optional")
    print("-" * 70)

# =============================================================================
# SECTION 6: VISUALIZATION
# =============================================================================

def plot_training_history(history):
    """
    Plots the training loss over epochs.
    """
    print("\n" + "=" * 70)
    print("Training History Visualization")
    print("=" * 70)
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], linewidth=2, label='Training Loss')
    plt.plot(history.history['val_loss'], linewidth=2, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mnist_training_loss.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved training loss plot to 'mnist_training_loss.png'")
    
    plt.show()

def visualize_predictions(model, X, y, num_samples=64):
    """
    Visualizes 64 random digit samples with their predicted vs actual labels.
    
    Creates an 8×8 grid showing:
    - Each digit image (20×20 pixels)
    - Title format: "Actual: X, Predicted: Y"
    - Green title if correct, red if incorrect
    """
    print("\n" + "=" * 70)
    print(f"Visualizing {num_samples} Random Predictions")
    print("=" * 70)
    
    # Get random indices
    indices = np.random.choice(len(X), num_samples, replace=False)
    
    # Get predictions for all samples
    logits_all = model.predict(X, verbose=0)
    predictions = np.argmax(logits_all, axis=1)
    
    # Create 8×8 grid
    fig, axes = plt.subplots(8, 8, figsize=(16, 16))
    fig.suptitle('MNIST Digit Recognition - Actual vs Predicted', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    for i, idx in enumerate(indices):
        row, col = i // 8, i % 8
        ax = axes[row, col]
        
        # Reshape to 20×20 image
        img = X[idx].reshape(20, 20)
        
        # Get labels
        actual = y[idx]
        predicted = predictions[idx]
        
        # Display image
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        
        # Set title with color coding
        title = f"Actual: {actual}, Pred: {predicted}"
        color = 'green' if actual == predicted else 'red'
        ax.set_title(title, fontsize=8, color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('mnist_predictions_grid.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved predictions grid to 'mnist_predictions_grid.png'")
    print(f"  ✓ Green titles: Correct predictions")
    print(f"  ✓ Red titles: Incorrect predictions")
    
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution pipeline.
    """
    # Step 1: Load data
    X, y = load_mnist_data()
    
    # Step 2: Test softmax implementations
    test_softmax_implementations()
    
    # Step 3: Build model
    model = build_model()
    
    # Step 4: Train model
    history = train_model(model, X, y, epochs=40)
    
    # Step 5: Demonstrate prediction
    demonstrate_prediction(model, X, y)
    
    # Step 6: Visualize results
    plot_training_history(history)
    visualize_predictions(model, X, y, num_samples=64)
    
    print("\n" + "=" * 70)
    print("✓ MNIST Multiclass Classification Complete!")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("Final Model Performance:")
    print("-" * 70)
    final_acc = history.history['accuracy'][-1] * 100
    final_val_acc = history.history['val_accuracy'][-1] * 100
    print(f"  Training Accuracy: {final_acc:.2f}%")
    print(f"  Validation Accuracy: {final_val_acc:.2f}%")
    
    print("\n" + "-" * 70)
    print("Key Takeaways:")
    print("-" * 70)
    print("1. Use linear output + from_logits=True for numerical stability")
    print("2. Softmax converts logits to probabilities, but argmax works directly")
    print("3. ReLU hidden layers create non-linear decision boundaries")
    print("4. MNIST is a challenging 10-class problem - our model performs well!")
    print("=" * 70)

if __name__ == "__main__":
    main()
