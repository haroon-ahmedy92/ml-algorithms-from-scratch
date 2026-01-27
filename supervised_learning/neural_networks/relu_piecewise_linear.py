"""
ReLU Piecewise Linear Approximation Experiment
===============================================

This script demonstrates how ReLU (Rectified Linear Unit) activation functions 
enable neural networks to approximate piecewise linear functions by "stitching" 
together linear segments.

Educational Objective:
----------------------
Understanding that a neural network with ReLU activations can represent ANY 
piecewise linear function by combining multiple "bent" linear units.

Mathematical Foundation:
------------------------
The ReLU activation function is defined as:
$$\text{ReLU}(z) = \max(0, z) = \begin{cases} 
z & \text{if } z > 0 \\
0 & \text{if } z \leq 0
\end{cases}$$

For a 2-layer network with ReLU in the hidden layer:
$$f(x) = W^{[2]} \cdot \text{ReLU}(W^{[1]} x + b^{[1]}) + b^{[2]}$$

Expanding for 3 hidden units:
$$f(x) = \sum_{i=0}^{2} w_i^{[2]} \cdot \max(0, w_i^{[1]} x + b_i^{[1]}) + b^{[2]}$$

Each ReLU unit can "turn on" in a specific region of the input space, 
creating a piecewise linear function.

Piecewise Linear Target:
-------------------------
We create a target function with 3 segments:
- Segment 1 (x ∈ [0, 1]): y = 2x (slope = 2)
- Segment 2 (x ∈ [1, 2]): y = 2 + 1(x-1) = 1 + x (slope = 1)
- Segment 3 (x ∈ [2, 3]): y = 3 + (-2)(x-2) = 7 - 2x (slope = -2)

Strategy:
---------
Each hidden unit is configured to activate in a specific region:
- Unit 0: Contributes slope 2 for x ∈ [0, 1]
- Unit 1: Contributes slope 1 for x ∈ [1, 2]  
- Unit 2: Contributes slope -2 for x ∈ [2, 3]

Author: ML Algorithms from Scratch Project
Date: January 27, 2026
License: MIT
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import os

# =============================================================================
# SECTION 1: DATA GENERATION
# =============================================================================

def create_piecewise_linear_data():
    """
    Creates a 1D synthetic dataset with a piecewise linear target.
    
    Piecewise Definition:
    ---------------------
    $$y(x) = \begin{cases}
    2x & \text{if } 0 \leq x < 1 \\
    1 + x & \text{if } 1 \leq x < 2 \\
    7 - 2x & \text{if } 2 \leq x \leq 3
    \end{cases}$$
    
    The function has "kinks" at x = 1 and x = 2 where the slope changes.
    
    Returns:
        X (ndarray): Shape (n, 1) - Input values
        y (ndarray): Shape (n,) - Piecewise linear target
    """
    print("=" * 70)
    print("ReLU Piecewise Linear Approximation Experiment")
    print("=" * 70)
    
    # Create dense sampling from 0 to 3
    X = np.linspace(0, 3, 300).reshape(-1, 1)
    y = np.zeros(X.shape[0])
    
    # Define the piecewise function
    for i, x_val in enumerate(X):
        x = x_val[0]
        if 0 <= x < 1:
            # Segment 1: slope 2
            y[i] = 2 * x
        elif 1 <= x < 2:
            # Segment 2: slope 1, starts at y=2 when x=1
            y[i] = 2 + 1 * (x - 1)
        else:
            # Segment 3: slope -2, starts at y=3 when x=2
            y[i] = 3 + (-2) * (x - 2)
    
    print("\n✓ Piecewise Linear Dataset Created:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  X range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"  y range: [{y.min():.2f}, {y.max():.2f}]")
    print("\n  Segment 1 (x∈[0,1]): y = 2x (slope=2)")
    print("  Segment 2 (x∈[1,2]): y = 1+x (slope=1)")
    print("  Segment 3 (x∈[2,3]): y = 7-2x (slope=-2)")
    
    return X, y

def save_and_load_data(X, y):
    """
    Saves the dataset to disk and loads it back (maintaining local-loading workflow).
    
    Parameters:
        X, y: Data arrays
        
    Returns:
        X, y: Loaded data arrays
    """
    # Create data directory
    os.makedirs('./data', exist_ok=True)
    
    # Save as a single file with both X and y
    data = {'X': X, 'y': y}
    np.save('./data/relu_data.npy', data)
    print("\n✓ Data saved to './data/relu_data.npy'")
    
    # Load it back
    loaded_data = np.load('./data/relu_data.npy', allow_pickle=True).item()
    X_loaded = loaded_data['X']
    y_loaded = loaded_data['y']
    
    print("✓ Data loaded from file")
    print(f"  X: {X_loaded.shape}, y: {y_loaded.shape}")
    
    return X_loaded, y_loaded

# =============================================================================
# SECTION 2: MODEL ARCHITECTURE WITH MANUAL WEIGHTS
# =============================================================================

def build_model_with_manual_weights():
    """
    Builds a 2-layer neural network and manually sets weights to approximate 
    the piecewise linear function.
    
    Architecture:
    -------------
    Input (1D) -> Layer 1 (3 units, ReLU) -> Layer 2 (1 unit, Linear) -> Output
    
    Weight Configuration Strategy:
    -------------------------------
    Layer 1 (W1, b1): Create 3 ReLU units that activate in different regions
    
    Unit 0: Handles x ∈ [0, 1]
      - W1[0] = slope, b1[0] = intercept
      - ReLU ensures it only contributes when x > 0
      
    Unit 1: Handles x ∈ [1, 2]
      - W1[1] = slope, b1[1] = -1 (turns on at x=1)
      - ReLU ensures it only contributes when x > 1
      
    Unit 2: Handles x ∈ [2, 3]
      - W1[2] = slope, b1[2] = -2 (turns on at x=2)
      - ReLU ensures it only contributes when x > 2
    
    Layer 2 (W2, b2): Combine the units
      - W2 weights determine how each unit contributes to final output
      - b2 is the overall bias/offset
    
    Mathematical Representation:
    ----------------------------
    $$a_0^{[1]} = \text{ReLU}(w_0^{[1]} x + b_0^{[1]})$$
    $$a_1^{[1]} = \text{ReLU}(w_1^{[1]} x + b_1^{[1]})$$
    $$a_2^{[1]} = \text{ReLU}(w_2^{[1]} x + b_2^{[1]})$$
    
    $$f(x) = \sum_{i=0}^{2} a_i^{[1]} w_i^{[2]} + b^{[2]}$$
    
    Returns:
        model: Configured Keras model with manual weights
    """
    print("\n" + "=" * 70)
    print("Building Model with Manual Weights")
    print("=" * 70)
    
    # Build the model
    model = Sequential([
        Dense(3, activation='relu', name='layer1'),
        Dense(1, activation='linear', name='layer2')
    ], name='relu_piecewise_model')
    
    # Build the model by passing a sample input
    model.build(input_shape=(None, 1))
    
    print("\nModel Architecture:")
    model.summary()
    
    # Manually set weights for Layer 1
    # Each unit creates a "hinge" at different x values
    
    # Unit 0: Active for x ∈ [0, 1], contributes slope 2
    # ReLU(2*x + 0) = 2*x for x > 0
    w1_0 = 2.0
    b1_0 = 0.0
    
    # Unit 1: Active for x ∈ [1, 2], contributes additional slope change
    # ReLU(1*x - 1) = x - 1 for x > 1
    # This changes slope from 2 to 1 (reduction of 1)
    w1_1 = 1.0
    b1_1 = -1.0
    
    # Unit 2: Active for x ∈ [2, 3], contributes additional slope change
    # ReLU(-2*x + 4) = -2*x + 4 for x < 2, but we want it to activate at x=2
    # Actually: ReLU(-3*(x-2)) activates for x > 2
    w1_2 = -3.0
    b1_2 = 6.0  # -3*2 + 6 = 0, so it starts at x=2
    
    # Combine into weight matrices
    # For Dense layer with input_shape=(1,) and 3 units, W1 should be (1, 3)
    W1 = np.array([[w1_0, w1_1, w1_2]])  # Shape (1, 3)
    b1 = np.array([b1_0, b1_1, b1_2])    # Shape (3,)
    
    # Set Layer 1 weights
    model.get_layer('layer1').set_weights([W1, b1])
    
    # Manually set weights for Layer 2
    # Simply sum all three units
    W2 = np.array([[1.0], [1.0], [1.0]])  # Shape (3, 1) - sum all units
    b2 = np.array([0.0])                   # Shape (1,)
    
    # Set Layer 2 weights
    model.get_layer('layer2').set_weights([W2, b2])
    
    print("\n" + "-" * 70)
    print("Layer 1 Weights (Creating ReLU Hinges):")
    print("-" * 70)
    print(f"Unit 0: W={w1_0:.2f}, b={b1_0:.2f} → Active for x > 0")
    print(f"Unit 1: W={w1_1:.2f}, b={b1_1:.2f} → Active for x > 1")
    print(f"Unit 2: W={w1_2:.2f}, b={b1_2:.2f} → Active for x > 2")
    
    print("\n" + "-" * 70)
    print("Layer 2 Weights (Combining Units):")
    print("-" * 70)
    print(f"W2 = {W2.flatten()} (sum all units)")
    print(f"b2 = {b2[0]:.2f}")
    
    print("\n" + "-" * 70)
    print("How It Works:")
    print("-" * 70)
    print("• For x ∈ [0,1]: Only Unit 0 is active → output = 2x")
    print("• For x ∈ [1,2]: Units 0&1 active → output = 2x + (x-1) = x+1")
    print("• For x ∈ [2,3]: All units active → output = 2x + (x-1) + (-3x+6) = 7-2x")
    print("-" * 70)
    
    return model

# =============================================================================
# SECTION 3: EXTRACT UNIT ACTIVATIONS
# =============================================================================

def extract_unit_activations(model, X):
    """
    Extracts the activations from each unit in Layer 1.
    
    This allows us to visualize how each ReLU unit contributes to the final output.
    
    Parameters:
        model: Trained Keras model
        X: Input data
        
    Returns:
        activations: Dictionary with 'unit_0', 'unit_1', 'unit_2', 'final_output'
    """
    print("\n" + "=" * 70)
    print("Extracting Unit Activations")
    print("=" * 70)
    
    # First, make a prediction to ensure the model is built
    _ = model.predict(X[:1], verbose=0)
    
    # Create a model that outputs Layer 1 activations
    layer1_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer('layer1').output
    )
    
    # Get activations for all units
    layer1_output = layer1_model.predict(X, verbose=0)
    
    # Get final output
    final_output = model.predict(X, verbose=0).flatten()
    
    activations = {
        'unit_0': layer1_output[:, 0],
        'unit_1': layer1_output[:, 1],
        'unit_2': layer1_output[:, 2],
        'final_output': final_output
    }
    
    print(f"  Unit 0 activations: {activations['unit_0'].shape}")
    print(f"  Unit 1 activations: {activations['unit_1'].shape}")
    print(f"  Unit 2 activations: {activations['unit_2'].shape}")
    print(f"  Final output: {activations['final_output'].shape}")
    
    return activations

# =============================================================================
# SECTION 4: VISUALIZATION
# =============================================================================

def plot_piecewise_approximation(X, y, activations):
    """
    Creates a 4-subplot visualization showing:
    1. Unit 0 output (ReLU applied)
    2. Unit 1 output (ReLU applied)
    3. Unit 2 output (ReLU applied)
    4. Final stitched prediction vs target
    
    This demonstrates how ReLU units combine to approximate piecewise linear functions.
    """
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    X_flat = X.flatten()
    
    # Subplot 1: Unit 0 (handles x ∈ [0, 1])
    axes[0, 0].plot(X_flat, activations['unit_0'], 'b-', linewidth=2, label='Unit 0 Output')
    axes[0, 0].axvline(x=1, color='red', linestyle='--', alpha=0.5, label='Activation boundary')
    axes[0, 0].set_xlabel('x', fontsize=12)
    axes[0, 0].set_ylabel('ReLU(W₀·x + b₀)', fontsize=12)
    axes[0, 0].set_title('Unit 0: Active for x ∈ [0, 1]', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Subplot 2: Unit 1 (handles x ∈ [1, 2])
    axes[0, 1].plot(X_flat, activations['unit_1'], 'g-', linewidth=2, label='Unit 1 Output')
    axes[0, 1].axvline(x=1, color='red', linestyle='--', alpha=0.5, label='Turn-on point')
    axes[0, 1].axvline(x=2, color='orange', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('x', fontsize=12)
    axes[0, 1].set_ylabel('ReLU(W₁·x + b₁)', fontsize=12)
    axes[0, 1].set_title('Unit 1: Active for x > 1', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Subplot 3: Unit 2 (handles x ∈ [2, 3])
    axes[1, 0].plot(X_flat, activations['unit_2'], 'purple', linewidth=2, label='Unit 2 Output')
    axes[1, 0].axvline(x=2, color='red', linestyle='--', alpha=0.5, label='Turn-on point')
    axes[1, 0].set_xlabel('x', fontsize=12)
    axes[1, 0].set_ylabel('ReLU(W₂·x + b₂)', fontsize=12)
    axes[1, 0].set_title('Unit 2: Active for x > 2', fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Subplot 4: Final stitched output vs target
    axes[1, 1].plot(X_flat, y, 'k--', linewidth=2, label='Target (Piecewise Linear)', alpha=0.7)
    axes[1, 1].plot(X_flat, activations['final_output'], 'r-', linewidth=2, label='Model Prediction')
    axes[1, 1].axvline(x=1, color='blue', linestyle='--', alpha=0.3)
    axes[1, 1].axvline(x=2, color='blue', linestyle='--', alpha=0.3)
    axes[1, 1].set_xlabel('x', fontsize=12)
    axes[1, 1].set_ylabel('y', fontsize=12)
    axes[1, 1].set_title('Final Output: Sum of ReLU Units', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.suptitle('ReLU Piecewise Linear Approximation\n' + 
                 r'$f(x) = \sum_{i=0}^{2} w_i^{[2]} \cdot \text{ReLU}(w_i^{[1]} x + b_i^{[1]}) + b^{[2]}$',
                 fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig('relu_piecewise_visualization.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved visualization to 'relu_piecewise_visualization.png'")
    
    plt.show()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main execution pipeline.
    """
    # Step 1: Create piecewise linear data
    X, y = create_piecewise_linear_data()
    
    # Step 2: Save and load data
    X, y = save_and_load_data(X, y)
    
    # Step 3: Build model with manual weights
    model = build_model_with_manual_weights()
    
    # Step 4: Extract unit activations
    activations = extract_unit_activations(model, X)
    
    # Step 5: Visualize the piecewise approximation
    plot_piecewise_approximation(X, y, activations)
    
    # Step 6: Compute approximation error
    mse = np.mean((y - activations['final_output'])**2)
    mae = np.mean(np.abs(y - activations['final_output']))
    
    print("\n" + "=" * 70)
    print("✓ ReLU Piecewise Linear Experiment Complete!")
    print("=" * 70)
    print(f"\nApproximation Quality:")
    print(f"  Mean Squared Error (MSE): {mse:.6f}")
    print(f"  Mean Absolute Error (MAE): {mae:.6f}")
    
    print("\n" + "-" * 70)
    print("Key Insights:")
    print("-" * 70)
    print("1. ReLU activation creates 'hinges' - points where the function bends")
    print("2. Each hidden unit activates in a specific region of input space")
    print("3. The output layer combines these activations to create piecewise segments")
    print("4. With enough ReLU units, we can approximate ANY piecewise linear function")
    print("5. This is why ReLU networks are powerful - they're 'universal approximators'")
    print("=" * 70)

if __name__ == "__main__":
    main()
