"""
Coffee Roasting - Manual Forward Propagation in NumPy
=====================================================

This script demonstrates how to implement Forward Propagation from scratch using NumPy,
effectively recreating the internal logic of a TensorFlow/Keras Dense layer.
This follows the "Neural Network Implementation in NumPy" lab from the Coursera ML Specialization.

Key Concepts:
-------------
1. **Forward Propagation**: Passing input data through the network to get predictions.
2. **Dense Layer**: A fully connected layer where every input connects to every unit.
3. **Activation Function**: Introducing non-linearity (Sigmoid in this case).
4. **Vectorization**: (Partially used here) Using dot products for efficiency.

Author: ML Algorithms from Scratch Project
Date: January 25, 2026
License: MIT
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Normalization

# =============================================================================
# SECTION 1: DATA & PREPROCESSING
# =============================================================================

def load_data():
    """
    Creates a synthetic dataset representing coffee roasting parameters.
    
    X: (200, 2) array.
       Column 0: Temperature (175-260)
       Column 1: Duration (12-15)
    Y: (200, 1) array. Labels (1=Good, 0=Bad).
    """
    rng = np.random.default_rng(2)
    X = rng.random((200, 2)) * [85, 3] + [175, 12]  # Scale and shift to range
    
    # Define a complex decision boundary for "Good" coffee
    # Good if distance from optimal center (200, 13.5) is small
    Y = np.zeros((200, 1))
    for i in range(len(X)):
        t = X[i, 0]
        d = X[i, 1]
        y_pos = -3/(260-175)*t + 21
        if (t > 175 and t < 260 and d > 12 and d < 15 and d < y_pos + 0.5 and d > y_pos - 0.5):
            Y[i, 0] = 1
        else:
            if rng.random() > 0.8: # Add some noise
                Y[i, 0] = 1

    print(f"Data Loaded: X shape {X.shape}, Y shape {Y.shape}")
    return X, Y

def normalize_data(X):
    """
    Normalizes features using TensorFlow's Normalization layer.
    Using Keras here ensures we match the normalization used when training
    the weights we will load later.
    
    Formula:
    $$x_{norm} = \frac{x - \mu}{\sigma}$$
    """
    print("\nNormalizing data...")
    norm_l = Normalization(axis=-1)
    norm_l.adapt(X)  # Learn mean and variance
    X_norm = norm_l(X)
    
    print(f"Normalization params:")
    print(f"  Mean: {norm_l.mean.numpy().flatten()}")
    print(f"  Variance: {norm_l.variance.numpy().flatten()}")
    return X_norm.numpy()

# =============================================================================
# SECTION 2: HELPER FUNCTIONS (Sigmoid)
# =============================================================================

def sigmoid(z):
    """
    Compute the sigmoid of z.
    
    Mathematical Formula:
    ---------------------
    $$g(z) = \frac{1}{1 + e^{-z}}$$
    
    Parameters:
        z (ndarray or scalar): Input value
        
    Returns:
        sigmoid(z)
    """
    # Clip z to avoid overflow in exp, though generally safe for typical NN weights
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

# =============================================================================
# SECTION 3: THE "MY_DENSE" FUNCTION
# =============================================================================

def my_dense(a_in, W, b, g):
    """
    Computes the output of a dense layer using a loop over units.
    
    Mathematical Logic:
    -------------------
    For a specific unit j in the layer:
    1. Compute the linear combination (dot product):
       $$z_j = \mathbf{w}_j \cdot \mathbf{a}_{in} + b_j$$
       
       Where:
       - $\mathbf{w}_j$ is the j-th column of input weights W.
       - $\mathbf{a}_{in}$ is the input vector from the previous layer.
       - $b_j$ is the bias for unit j.
       
    2. Apply activation function:
       $$a_j = g(z_j)$$
    
    The final output vector for the layer with n units is:
    $$\mathbf{a}_{out} = [g(z_0), g(z_1), ... g(z_{n-1})]$$
    
    Parameters:
        a_in (ndarray): Shape (n_in,) - Input vector
        W (ndarray): Shape (n_in, n_units) - Weight matrix
        b (ndarray): Shape (n_units,) - Bias vector
        g (function): Activation function (e.g., sigmoid)
        
    Returns:
        a_out (ndarray): Shape (n_units,) - Output vector
    """
    units = W.shape[1]
    a_out = np.zeros(units)
    
    # Iterate through each unit (neuron) in the layer
    for j in range(units):
        w = W[:, j]         # Select weights for unit j (column j)
        z = np.dot(w, a_in) + b[j]  # Scalar dot product + bias
        a_out[j] = g(z)     # Apply activation
        
    return a_out

# =============================================================================
# SECTION 4: THE SEQUENTIAL & PREDICT FUNCTIONS
# =============================================================================

def my_sequential(x, W1, b1, W2, b2):
    """
    Forward propagates input x through a 2-layer network.
    
    Architecture:
    -------------
    Input -> Layer 1 (Dense) -> Layer 2 (Dense) -> Output
    
    Parameters:
        x (ndarray): Input vector
        W1, b1: Weights/Biases for Layer 1
        W2, b2: Weights/Biases for Layer 2
        
    Returns:
        f_x (scalar): Final prediction probability
    """
    # Layer 1 Data flow
    a1 = my_dense(x, W1, b1, sigmoid)
    
    # Layer 2 Data flow
    a2 = my_dense(a1, W2, b2, sigmoid)
    
    return a2

def my_predict(X, W1, b1, W2, b2):
    """
    Iterates through all training examples to generate predictions.
    
    Process:
    --------
    Loop though each row (example) in X, apply my_sequential, 
    and collect results.
    
    Parameters:
        X (ndarray): Shape (m, n_features) - Dataset
        W1, b1, W2, b2: Model parameters
        
    Returns:
        predictions (ndarray): Shape (m, 1) - Predicted probabilities
    """
    m = X.shape[0]
    p = np.zeros((m, 1))
    
    for i in range(m):
        # Forward prop for a single example i
        # my_sequential returns an array [prob], so we extract the scalar
        p[i, 0] = my_sequential(X[i], W1, b1, W2, b2)[0]
        
    return p

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    print("=" * 70)
    print("Manual Neural Network Implementation - Coffee Roasting Lab")
    print("=" * 70)

    # 1. Load and Normalize Data
    X, Y = load_data()
    X_norm = normalize_data(X)

    # 2. Load Weights (Hardcoded from Lab)
    print("\nLoading pre-trained weights...")
    W1_tmp = np.array([[-8.93,  0.29, 12.9 ], 
                       [-0.1,  -7.32, 10.81]])
    b1_tmp = np.array([-9.82, -9.28,  0.96])
    
    W2_tmp = np.array([[-31.18], 
                       [-27.59], 
                       [-32.56]])
    b2_tmp = np.array([15.41])

    print("W1 shape:", W1_tmp.shape)
    print("b1 shape:", b1_tmp.shape)
    print("W2 shape:", W2_tmp.shape)
    print("b2 shape:", b2_tmp.shape)

    # 3. Make Predictions
    print("\nRunning inference (Forward Propagation) on all examples...")
    predictions = my_predict(X_norm, W1_tmp, b1_tmp, W2_tmp, b2_tmp)

    # 4. Apply Threshold
    # Convert probabilities to binary: 1 if >= 0.5, else 0
    yhat = (predictions >= 0.5).astype(int)
    
    # 5. Review Results
    # Filter predictions: Good Roast (1) vs Bad Roast (0)
    indices_good = np.where(predictions >= 0.5)[0]
    indices_bad = np.where(predictions < 0.5)[0]
    
    print(f"\nPredictions Summary:")
    print(f"  Total examples: {len(X)}")
    print(f"  Predicted Good Roasts: {len(indices_good)}")
    print(f"  Predicted Bad Roasts:  {len(indices_bad)}")
    
    # Compare with ground truth Y (just for validation)
    accuracy = np.mean(yhat == Y)
    print(f"  Accuracy vs Synthetic Labels: {accuracy * 100:.2f}%")

    # Show a few specific examples
    print("\nSample Predictions:")
    print(f"{'Input (Temp, Dur)':<25} | {'Probability':<12} | {'Prediction':<10}")
    print("-" * 55)
    for i in range(5): 
        idx = np.random.randint(0, 200)
        norm_val = X_norm[idx]
        orig_val = X[idx]
        prob = predictions[idx, 0]
        decision = "Good" if prob >= 0.5 else "Bad"
        print(f"[{orig_val[0]:.1f}, {orig_val[1]:.1f}]           | {prob:.4f}       | {decision}")

if __name__ == "__main__":
    main()
