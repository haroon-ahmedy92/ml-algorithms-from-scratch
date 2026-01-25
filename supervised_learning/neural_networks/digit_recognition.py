"""
Handwritten Digit Recognition (Binary Classification: 0 and 1)
===============================================================

This script demonstrates a complete neural network implementation for binary digit classification
using three distinct approaches:
1. TensorFlow/Keras Sequential API (High-level)
2. NumPy with FOR LOOPS (Educational, step-by-step)
3. NumPy with VECTORIZATION (Efficient, matrix operations)

Dataset:
--------
- X: (1000, 400) - Each row is a 20x20 pixel image flattened into a 400-dimensional vector
- y: (1000, 1)   - Binary labels: 0 or 1

Network Architecture:
---------------------
Layer 1: 400 inputs -> 25 units (Sigmoid)
Layer 2: 25 inputs  -> 15 units (Sigmoid)
Layer 3: 15 inputs  -> 1 unit   (Sigmoid)

Mathematical Foundation:
------------------------
For a single example, forward propagation through a dense layer:
$$z^{[l]} = W^{[l]} \cdot a^{[l-1]} + b^{[l]}$$
$$a^{[l]} = g(z^{[l]})$$

For vectorized operations with m examples:
$$Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}$$
where A is (m, n) and W is (n, j), resulting in Z of shape (m, j).

Author: ML Algorithms from Scratch Project
Date: January 25, 2026
License: MIT
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# =============================================================================
# SECTION 1: DATASET SETUP
# =============================================================================

def load_digit_data():
    """
    Loads handwritten digit dataset from .npy files and filters for binary classification (0 and 1).
    
    Data Structure:
    ---------------
    - X: (1000, 400) array representing 1000 images
      Each image is 20x20 pixels unrolled into a 400-dimensional vector.
      
      Mathematical Representation:
      $$X_{i} = [p_{0,0}, p_{0,1}, ..., p_{0,19}, p_{1,0}, ..., p_{19,19}]$$
      
      Where $p_{r,c}$ is the pixel at row r, column c.
      
    - y: (1000, 1) array of binary labels (0 or 1)
    
    Returns:
        X (ndarray): Shape (1000, 400) - Pixel intensities [0, 1]
        y (ndarray): Shape (1000, 1) - Binary labels
    """
    print("=" * 70)
    print("Handwritten Digit Recognition - Binary Classification (0 vs 1)")
    print("=" * 70)
    
    # Load data from .npy files
    print("\nLoading data from files...")
    X_full = np.load('data/X.npy')
    y_full = np.load('data/y.npy')
    
    print(f"Full dataset loaded:")
    print(f"  X shape: {X_full.shape}")
    print(f"  y shape: {y_full.shape}")
    print(f"  Unique labels: {np.unique(y_full.flatten())}")
    
    # Filter for binary classification: Only keep digits 0 and 1
    mask = np.isin(y_full.flatten(), [0, 1])
    X = X_full[mask]
    y = y_full[mask]
    
    print(f"\nFiltered Dataset (0 and 1 only):")
    print(f"  X shape: {X.shape} (images, each 20x20=400 pixels)")
    print(f"  y shape: {y.shape} (Binary labels: 0 or 1)")
    print(f"  Class 0 count: {np.sum(y == 0)}")
    print(f"  Class 1 count: {np.sum(y == 1)}")
    
    return X, y

def visualize_samples(X, y, num_samples=5):
    """
    Visualizes random samples from the dataset as 20x20 grayscale images.
    
    Pixel Unrolling:
    ----------------
    A 20x20 image is stored as a 400-dimensional vector:
    $$\text{Image}_{20 \times 20} \rightarrow \text{Vector}_{400 \times 1}$$
    
    To reconstruct: Reshape vector back to (20, 20) matrix.
    
    Parameters:
        X (ndarray): Shape (m, 400) - Flattened images
        y (ndarray): Shape (m, 1) - Labels
        num_samples (int): Number of random samples to display
    """
    print(f"\nVisualizing {num_samples} random samples...")
    
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 3))
    indices = np.random.choice(len(X), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        # Reshape 400-dim vector back to 20x20 image
        img = X[idx].reshape(20, 20)
        label = int(y[idx, 0])
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('digit_samples.png', dpi=100, bbox_inches='tight')
    print(f"  Saved visualization to 'digit_samples.png'")
    plt.close()

# =============================================================================
# SECTION 2: TENSORFLOW IMPLEMENTATION (Exercise 1)
# =============================================================================

def build_tensorflow_model():
    """
    Constructs a 3-layer neural network using TensorFlow/Keras Sequential API.
    
    Architecture:
    -------------
    Input (400) -> Dense(25, sigmoid) -> Dense(15, sigmoid) -> Dense(1, sigmoid)
    
    Weight Dimensions:
    ------------------
    If layer l-1 has n units and layer l has j units, then:
    $$W^{[l]} \in \mathbb{R}^{n \times j}$$
    $$b^{[l]} \in \mathbb{R}^{j}$$
    
    Example:
    - Layer 1: Input=400, Output=25 → W1 is (400, 25), b1 is (25,)
    - Layer 2: Input=25,  Output=15 → W2 is (25, 15),  b2 is (15,)
    - Layer 3: Input=15,  Output=1  → W3 is (15, 1),   b3 is (1,)
    
    Returns:
        model (Sequential): Compiled Keras model
    """
    print("\n" + "=" * 70)
    print("EXERCISE 1: TensorFlow/Keras Implementation")
    print("=" * 70)
    
    my_model = Sequential([
        Dense(units=25, activation='sigmoid', name='layer1'),
        Dense(units=15, activation='sigmoid', name='layer2'),
        Dense(units=1,  activation='sigmoid', name='layer3')
    ], name='my_model')
    
    # Compile with Binary Cross-Entropy Loss
    # Loss Function:
    # $$L = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)})]$$
    my_model.compile(
        loss=BinaryCrossentropy(),
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    print("\nModel Architecture:")
    my_model.summary()
    
    return my_model

def train_tensorflow_model(model, X, y, epochs=20):
    """
    Trains the TensorFlow model using the provided dataset.
    
    Training Process:
    -----------------
    For each epoch, the optimizer minimizes the loss function by updating weights:
    $$W := W - \alpha \frac{\partial L}{\partial W}$$
    
    where $\alpha$ is the learning rate (0.001).
    
    Parameters:
        model (Sequential): The Keras model
        X (ndarray): Training features
        y (ndarray): Training labels
        epochs (int): Number of training iterations
        
    Returns:
        history: Training history object
    """
    print(f"\nTraining for {epochs} epochs...")
    
    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    
    print("\n✓ Training Complete!")
    final_loss = history.history['loss'][-1]
    final_acc = history.history['accuracy'][-1]
    print(f"  Final Loss: {final_loss:.4f}")
    print(f"  Final Accuracy: {final_acc * 100:.2f}%")
    
    return history

# =============================================================================
# SECTION 3: NUMPY LOOP IMPLEMENTATION (Exercise 2)
# =============================================================================

def sigmoid(z):
    """
    Compute the sigmoid activation function.
    
    Formula:
    --------
    $$g(z) = \frac{1}{1 + e^{-z}}$$
    
    Properties:
    - Output range: (0, 1)
    - Derivative: $g'(z) = g(z)(1 - g(z))$
    
    Parameters:
        z (ndarray or scalar): Input value(s)
        
    Returns:
        sigmoid(z)
    """
    z = np.clip(z, -500, 500)  # Prevent overflow
    return 1.0 / (1.0 + np.exp(-z))

def my_dense(a_in, W, b, g):
    """
    Computes dense layer output using FOR LOOP over units (educational version).
    
    Mathematical Process:
    ---------------------
    For each unit j in layer l:
    
    1. Extract weight vector for unit j:
       $$\mathbf{w}_j = W[:, j]$$
       
    2. Compute weighted sum (linear combination):
       $$z_j = \mathbf{w}_j^T \mathbf{a}_{in} + b_j = \sum_{i=1}^{n} w_{j,i} \cdot a_{in,i} + b_j$$
       
    3. Apply activation:
       $$a_j = g(z_j)$$
    
    Dimension Analysis:
    -------------------
    - a_in: (n,) - Input from previous layer (n units)
    - W: (n, j) - Weight matrix (n inputs, j outputs)
    - b: (j,) - Bias vector (one per output unit)
    - a_out: (j,) - Output activations
    
    Parameters:
        a_in (ndarray): Input vector, shape (n,)
        W (ndarray): Weight matrix, shape (n, units)
        b (ndarray): Bias vector, shape (units,)
        g (function): Activation function
        
    Returns:
        a_out (ndarray): Output vector, shape (units,)
    """
    units = W.shape[1]  # Number of output units
    a_out = np.zeros(units)
    
    # Loop through each unit
    for j in range(units):
        w = W[:, j]  # Weight vector for unit j
        z = np.dot(w, a_in) + b[j]  # Linear combination
        a_out[j] = g(z)  # Activation
    
    return a_out

def my_sequential(x, W1, b1, W2, b2, W3, b3):
    r"""
    Forward propagates input through a 3-layer network using loops.
    
    Network Flow:
    -------------
    Layer 1 -> Layer 2 -> Layer 3
    
    Where each transformation is:
    a^{[l]} = g(W^{[l]T} a^{[l-1]} + b^{[l]})
    
    Parameters:
        x (ndarray): Input vector (400,)
        W1, b1: Layer 1 parameters (400, 25), (25,)
        W2, b2: Layer 2 parameters (25, 15), (15,)
        W3, b3: Layer 3 parameters (15, 1), (1,)
        
    Returns:
        prediction (scalar): Final output probability
    """
    a1 = my_dense(x, W1, b1, sigmoid)      # (400,) -> (25,)
    a2 = my_dense(a1, W2, b2, sigmoid)     # (25,) -> (15,)
    a3 = my_dense(a2, W3, b3, sigmoid)     # (15,) -> (1,)
    
    return a3[0]  # Extract scalar

def predict_loop(X, W1, b1, W2, b2, W3, b3):
    """
    Makes predictions for all examples using loop-based forward propagation.
    
    Process:
    --------
    For each example in the dataset, apply my_sequential function.
    
    Parameters:
        X (ndarray): Dataset, shape (m, 400)
        W1, b1, W2, b2, W3, b3: Network parameters
        
    Returns:
        predictions (ndarray): Shape (m, 1)
    """
    m = X.shape[0]
    predictions = np.zeros((m, 1))
    
    for i in range(m):
        predictions[i, 0] = my_sequential(X[i], W1, b1, W2, b2, W3, b3)
    
    return predictions

# =============================================================================
# SECTION 4: VECTORIZED NUMPY IMPLEMENTATION (Exercise 3)
# =============================================================================

def my_dense_v(A_in, W, b, g):
    """
    Computes dense layer output using VECTORIZED matrix multiplication.
    
    Vectorized Forward Propagation:
    --------------------------------
    Instead of looping through examples, process all m examples simultaneously:
    
    $$Z = A_{in} W + b$$
    
    where:
    - $A_{in}$: (m, n) - m examples, each with n features
    - $W$: (n, j) - Weight matrix
    - $b$: (j,) - Bias vector (broadcasted across all m examples)
    - $Z$: (m, j) - Linear outputs for all examples
    
    Then apply activation:
    $$A_{out} = g(Z)$$
    
    Dimension Verification:
    -----------------------
    $(m, n) \times (n, j) = (m, j)$
    
    Example: 1000 examples, Layer 1 (400 -> 25):
    $(1000, 400) \times (400, 25) = (1000, 25)$
    
    Parameters:
        A_in (ndarray): Input matrix, shape (m, n)
        W (ndarray): Weight matrix, shape (n, j)
        b (ndarray): Bias vector, shape (j,)
        g (function): Activation function
        
    Returns:
        A_out (ndarray): Output matrix, shape (m, j)
    """
    # Matrix multiplication: (m, n) @ (n, j) = (m, j)
    Z = np.matmul(A_in, W) + b  # Bias broadcasting
    A_out = g(Z)  # Element-wise activation
    
    return A_out

def my_sequential_v(X, W1, b1, W2, b2, W3, b3):
    """
    Vectorized forward propagation through 3-layer network.
    
    Processes ALL examples in parallel:
    $$A^{[1]} = g(X W^{[1]} + b^{[1]})$$
    $$A^{[2]} = g(A^{[1]} W^{[2]} + b^{[2]})$$
    $$\hat{Y} = g(A^{[2]} W^{[3]} + b^{[3]})$$
    
    Parameters:
        X (ndarray): Input matrix, shape (m, 400)
        W1, b1, W2, b2, W3, b3: Network parameters
        
    Returns:
        predictions (ndarray): Shape (m, 1)
    """
    A1 = my_dense_v(X, W1, b1, sigmoid)     # (m, 400) -> (m, 25)
    A2 = my_dense_v(A1, W2, b2, sigmoid)    # (m, 25) -> (m, 15)
    A3 = my_dense_v(A2, W3, b3, sigmoid)    # (m, 15) -> (m, 1)
    
    return A3

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # STEP 1: Load and Visualize Data
    # -------------------------------------------------------------------------
    X, y = load_digit_data()
    visualize_samples(X, y, num_samples=5)
    
    # -------------------------------------------------------------------------
    # STEP 2: TensorFlow Implementation (Exercise 1)
    # -------------------------------------------------------------------------
    tf_model = build_tensorflow_model()
    history = train_tensorflow_model(tf_model, X, y, epochs=20)
    
    # Make predictions
    print("\nTensorFlow Model Predictions (first 5 examples):")
    tf_predictions = tf_model.predict(X[:5])
    for i in range(5):
        print(f"  Example {i}: True={int(y[i, 0])}, Predicted={tf_predictions[i, 0]:.4f}")
    
    # -------------------------------------------------------------------------
    # STEP 3: NumPy Loop Implementation (Exercise 2)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXERCISE 2: NumPy Loop Implementation")
    print("=" * 70)
    
    # Extract weights from trained TensorFlow model
    W1, b1 = tf_model.layers[0].get_weights()
    W2, b2 = tf_model.layers[1].get_weights()
    W3, b3 = tf_model.layers[2].get_weights()
    
    print(f"\nExtracted Weights from TensorFlow model:")
    print(f"  W1: {W1.shape}, b1: {b1.shape}")
    print(f"  W2: {W2.shape}, b2: {b2.shape}")
    print(f"  W3: {W3.shape}, b3: {b3.shape}")
    
    # Test loop-based prediction on first example
    print("\nTesting loop-based forward propagation (first example):")
    loop_pred = my_sequential(X[0], W1, b1, W2, b2, W3, b3)
    print(f"  NumPy Loop Prediction: {loop_pred:.4f}")
    print(f"  TensorFlow Prediction: {tf_predictions[0, 0]:.4f}")
    print(f"  Match: {np.isclose(loop_pred, tf_predictions[0, 0])}")
    
    # -------------------------------------------------------------------------
    # STEP 4: Vectorized NumPy Implementation (Exercise 3)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXERCISE 3: Vectorized NumPy Implementation")
    print("=" * 70)
    
    # Test vectorized prediction on all examples
    print("\nTesting vectorized forward propagation (all 1000 examples):")
    vectorized_preds = my_sequential_v(X, W1, b1, W2, b2, W3, b3)
    
    print(f"  Vectorized predictions shape: {vectorized_preds.shape}")
    print(f"\nFirst 5 predictions comparison:")
    print(f"  {'TensorFlow':<15} {'NumPy Loop':<15} {'NumPy Vectorized':<20}")
    print("-" * 55)
    
    loop_preds_sample = []
    for i in range(5):
        loop_p = my_sequential(X[i], W1, b1, W2, b2, W3, b3)
        loop_preds_sample.append(loop_p)
        tf_p = tf_predictions[i, 0]
        vec_p = vectorized_preds[i, 0]
        print(f"  {tf_p:.6f}      {loop_p:.6f}      {vec_p:.6f}")
    
    # Verify all three methods produce identical results
    all_loop_preds = predict_loop(X, W1, b1, W2, b2, W3, b3)
    all_tf_preds = tf_model.predict(X, verbose=0)
    
    match_loop = np.allclose(all_loop_preds, all_tf_preds, atol=1e-5)
    match_vec = np.allclose(vectorized_preds, all_tf_preds, atol=1e-5)
    
    print(f"\n✓ Verification Results:")
    print(f"  NumPy Loop matches TensorFlow: {match_loop}")
    print(f"  NumPy Vectorized matches TensorFlow: {match_vec}")
    
    # Final accuracy check
    final_preds = (vectorized_preds >= 0.5).astype(int)
    accuracy = np.mean(final_preds == y)
    print(f"\n✓ Final Model Performance:")
    print(f"  Accuracy on training set: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
