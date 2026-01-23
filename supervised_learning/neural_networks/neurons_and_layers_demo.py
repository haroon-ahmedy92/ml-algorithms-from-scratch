"""
Neurons and Layers - TensorFlow/Keras Implementation
====================================================

This script demonstrates the inner workings of Neurons and Layers using TensorFlow/Keras,
recreating the "Neurons and Layers" lab from the Coursera ML Specialization.

Author: ML Algorithms from Scratch Project
Date: January 22, 2026
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.activations import sigmoid


# =============================================================================
# SECTION 1: LINEAR REGRESSION (Neuron without Activation)
# =============================================================================

def section1_linear_regression():
    """
    Demonstrate a single neuron performing linear regression.
    
    Mathematical Formula:
    ---------------------
    f_{w,b}(x) = wx + b
    
    Where:
        - w: weight parameter
        - b: bias parameter
        - x: input feature
        - f_{w,b}(x): linear output (no activation function)
    
    This represents a neuron with LINEAR activation (identity function).
    """
    print("=" * 80)
    print("SECTION 1: LINEAR REGRESSION - NEURON WITHOUT ACTIVATION")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Step 1: Define Training Data
    # -------------------------------------------------------------------------
    print("\n📊 Step 1: Define Training Data")
    print("-" * 80)
    
    # Training data for house price prediction example
    # X: size in 1000 sqft, Y: price in 1000s of dollars
    X_train = np.array([[1.0], [2.0]])  # Shape: (2, 1)
    Y_train = np.array([[300.0], [500.0]])  # Shape: (2, 1)
    
    print(f"Input features (X_train):\n{X_train}")
    print(f"Shape: {X_train.shape}")
    print(f"\nTarget values (Y_train):\n{Y_train}")
    print(f"Shape: {Y_train.shape}")
    
    # -------------------------------------------------------------------------
    # Step 2: Create Keras Model with Dense Layer
    # -------------------------------------------------------------------------
    print("\n🧠 Step 2: Create Keras Model with Linear Activation")
    print("-" * 80)
    
    """
    Dense Layer Configuration:
    --------------------------
    - units=1: One neuron (output dimension = 1)
    - activation='linear': f(x) = x (identity/no activation)
    - input_dim=1: One input feature
    
    The Dense layer implements:
        output = activation(dot(input, weights) + bias)
    
    For linear activation:
        f_{w,b}(x) = wx + b
    """
    
    # Create a Sequential model with one Dense layer
    model = Sequential([
        Dense(units=1, activation='linear', input_shape=(1,))
    ])
    
    # Alternative way using Sequential:
    # model = Sequential()
    # model.add(Input(shape=(1,)))
    # model.add(Dense(1, activation='linear'))
    
    print("✓ Created Sequential model:")
    print(f"  - Architecture: Dense(1, linear)")
    print(f"  - Input shape: (1,)")
    print(f"  - Output: Single continuous value")
    
    # Display model architecture
    print("\nModel Summary:")
    model.summary()
    
    # -------------------------------------------------------------------------
    # Step 3: Set Weights and Bias Manually
    # -------------------------------------------------------------------------
    print("\n⚙️  Step 3: Set Weights and Bias Manually")
    print("-" * 80)
    
    """
    Weight Initialization:
    ----------------------
    w = 200  (weight parameter)
    b = 100  (bias parameter)
    
    This means the model will compute:
        f_{w,b}(x) = 200x + 100
    
    For example:
        - x = 1: f(1) = 200(1) + 100 = 300
        - x = 2: f(2) = 200(2) + 100 = 500
    """
    
    w = 200.0
    b = 100.0
    
    # Set weights for the Dense layer
    # Weights: (input_dim, units) = (1, 1)
    # Bias: (units,) = (1,)
    model.set_weights([np.array([[w]]), np.array([b])])
    
    print(f"Set weights manually:")
    print(f"  w = {w} (weight)")
    print(f"  b = {b} (bias)")
    print(f"\nMathematical formula: f(x) = {w}x + {b}")
    
    # Verify weights
    weights, bias = model.get_weights()
    print(f"\nVerification - Retrieved weights:")
    print(f"  Weight matrix: {weights.flatten()}")
    print(f"  Bias vector: {bias}")
    
    # -------------------------------------------------------------------------
    # Step 4: Make Predictions using Keras
    # -------------------------------------------------------------------------
    print("\n🎯 Step 4: Make Predictions using Keras")
    print("-" * 80)
    
    # Make predictions using the Keras model
    Y_pred_keras = model.predict(X_train, verbose=0)
    
    print("Keras predictions:")
    for i, (x, y_pred) in enumerate(zip(X_train, Y_pred_keras)):
        print(f"  x = {x[0]:.1f}: f(x) = {y_pred[0]:.1f}")
    
    # -------------------------------------------------------------------------
    # Step 5: Manual Calculation using NumPy
    # -------------------------------------------------------------------------
    print("\n🔍 Step 5: Manual Calculation using NumPy (Verification)")
    print("-" * 80)
    
    """
    Manual calculation of linear regression:
    ----------------------------------------
    f_{w,b}(x) = wx + b
    
    Using matrix notation:
        Y = X @ W + b
    
    Where:
        @ is the matrix multiplication operator (dot product)
        X: (m, n) input matrix
        W: (n, 1) weight matrix
        b: (1,) bias vector
        Y: (m, 1) output matrix
    """
    
    # Manual computation using NumPy
    W_manual = np.array([[w]])  # Shape: (1, 1)
    b_manual = np.array([b])    # Shape: (1,)
    
    # Matrix multiplication: X @ W + b
    Y_pred_manual = np.dot(X_train, W_manual) + b_manual
    
    print("Manual NumPy predictions:")
    for i, (x, y_pred) in enumerate(zip(X_train, Y_pred_manual)):
        print(f"  x = {x[0]:.1f}: f(x) = {w}*{x[0]:.1f} + {b} = {y_pred[0]:.1f}")
    
    # -------------------------------------------------------------------------
    # Step 6: Compare Keras vs Manual Predictions
    # -------------------------------------------------------------------------
    print("\n📊 Step 6: Comparison - Keras vs Manual Predictions")
    print("-" * 80)
    
    print(f"\n{'X':<10} {'Keras':<15} {'Manual':<15} {'Match?':<10}")
    print("-" * 50)
    
    for i in range(len(X_train)):
        keras_val = Y_pred_keras[i][0]
        manual_val = Y_pred_manual[i][0]
        match = "✓ Yes" if np.allclose(keras_val, manual_val) else "✗ No"
        print(f"{X_train[i][0]:<10.1f} {keras_val:<15.1f} {manual_val:<15.1f} {match:<10}")
    
    # Verify all predictions match
    all_match = np.allclose(Y_pred_keras, Y_pred_manual)
    print(f"\n{'='*50}")
    print(f"✅ All predictions match: {all_match}")
    print(f"{'='*50}")
    
    # -------------------------------------------------------------------------
    # Step 7: Visualization
    # -------------------------------------------------------------------------
    print("\n📈 Step 7: Visualizing Linear Regression")
    print("-" * 80)
    
    plt.figure(figsize=(10, 6))
    
    # Plot training data
    plt.scatter(X_train, Y_train, color='red', s=100, marker='x', 
                linewidth=2, label='Training Data', zorder=3)
    
    # Plot predictions
    plt.scatter(X_train, Y_pred_keras, color='blue', s=100, 
                marker='o', alpha=0.6, label='Keras Predictions', zorder=2)
    
    # Plot regression line
    x_line = np.linspace(0, 3, 100).reshape(-1, 1)
    y_line = model.predict(x_line, verbose=0)
    plt.plot(x_line, y_line, 'g-', linewidth=2, 
             label=f'f(x) = {w}x + {b}', zorder=1)
    
    plt.xlabel('Size (1000 sqft)', fontsize=12)
    plt.ylabel('Price (1000s of dollars)', fontsize=12)
    plt.title('Linear Regression - Single Neuron (No Activation)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return model, Y_pred_keras, Y_pred_manual


# =============================================================================
# SECTION 2: LOGISTIC REGRESSION (Neuron with Sigmoid Activation)
# =============================================================================

def section2_logistic_regression():
    """
    Demonstrate a single neuron performing logistic regression.
    
    Mathematical Formula:
    ---------------------
    f_{w,b}(x) = g(wx + b)
    
    where g(z) is the sigmoid function:
        g(z) = 1 / (1 + e^{-z})
    
    Combined:
        f_{w,b}(x) = 1 / (1 + e^{-(wx + b)})
    
    Where:
        - w: weight parameter
        - b: bias parameter
        - x: input feature
        - z = wx + b: linear combination
        - g(z): sigmoid activation function
        - f_{w,b}(x): probability output (0 to 1)
    
    This represents a neuron with SIGMOID activation for binary classification.
    """
    print("\n" + "=" * 80)
    print("SECTION 2: LOGISTIC REGRESSION - NEURON WITH SIGMOID ACTIVATION")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Step 1: Define Training Data
    # -------------------------------------------------------------------------
    print("\n📊 Step 1: Define Training Data")
    print("-" * 80)
    
    # Binary classification example
    # X: hours studied, Y: pass (1) or fail (0)
    X_train = np.array([0., 1., 2., 3., 4., 5.]).reshape(-1, 1)  # Shape: (6, 1)
    Y_train = np.array([0., 0., 0., 1., 1., 1.]).reshape(-1, 1)  # Shape: (6, 1)
    
    print(f"Input features (hours studied):\n{X_train.T}")
    print(f"Shape: {X_train.shape}")
    print(f"\nTarget labels (pass/fail):\n{Y_train.T}")
    print(f"Shape: {Y_train.shape}")
    print(f"\nClass distribution:")
    print(f"  Class 0 (fail): {np.sum(Y_train == 0):.0f} examples")
    print(f"  Class 1 (pass): {np.sum(Y_train == 1):.0f} examples")
    
    # -------------------------------------------------------------------------
    # Step 2: Create Keras Sequential Model
    # -------------------------------------------------------------------------
    print("\n🧠 Step 2: Create Keras Sequential Model with Sigmoid Activation")
    print("-" * 80)
    
    """
    Sequential Model with Dense Layer:
    ----------------------------------
    - Dense(units=1): Single neuron output
    - activation='sigmoid': g(z) = 1/(1 + e^{-z})
    - input_shape=(1,): One input feature
    
    The model computes:
        1. Linear combination: z = wx + b
        2. Sigmoid activation: a = g(z) = 1/(1 + e^{-z})
        3. Output: probability between 0 and 1
    
    Full formula:
        f_{w,b}(x) = 1 / (1 + e^{-(wx + b)})
    """
    
    # Create Sequential model
    model = Sequential([
        Dense(units=1, activation='sigmoid', input_shape=(1,))
    ])
    
    # Alternative way to build the model:
    # model = Sequential()
    # model.add(Input(shape=(1,)))
    # model.add(Dense(1, activation='sigmoid'))
    
    print("✓ Created Sequential model:")
    print(f"  - Architecture: Dense(1, sigmoid)")
    print(f"  - Input shape: (1,)")
    print(f"  - Output: Single probability value (0 to 1)")
    
    # Display model architecture
    print("\nModel Summary:")
    model.summary()
    
    # -------------------------------------------------------------------------
    # Step 3: Set Weights and Bias Manually
    # -------------------------------------------------------------------------
    print("\n⚙️  Step 3: Set Weights and Bias Manually")
    print("-" * 80)
    
    """
    Weight Initialization:
    ----------------------
    w = 2.0   (weight parameter)
    b = -4.5  (bias parameter)
    
    The model will compute:
        z = 2.0 * x - 4.5
        f_{w,b}(x) = 1 / (1 + e^{-z})
    
    Decision boundary (where f(x) = 0.5):
        2.0 * x - 4.5 = 0
        x = 4.5 / 2.0 = 2.25
    
    Interpretation:
        - If x < 2.25: f(x) < 0.5 → predict class 0
        - If x > 2.25: f(x) > 0.5 → predict class 1
    """
    
    w = 2.0
    b = -4.5
    
    # Set weights for the Dense layer
    # Weights: (input_dim, units) = (1, 1)
    # Bias: (units,) = (1,)
    model.set_weights([np.array([[w]]), np.array([b])])
    
    print(f"Set weights manually:")
    print(f"  w = {w} (weight)")
    print(f"  b = {b} (bias)")
    print(f"\nMathematical formula:")
    print(f"  z = {w}x + ({b}) = {w}x - {abs(b)}")
    print(f"  f(x) = 1 / (1 + e^(-z))")
    print(f"\nDecision boundary (f(x) = 0.5):")
    print(f"  x = {abs(b)/w:.2f} hours")
    
    # Verify weights
    weights, bias = model.get_weights()
    print(f"\nVerification - Retrieved weights:")
    print(f"  Weight matrix: {weights.flatten()}")
    print(f"  Bias vector: {bias}")
    
    # -------------------------------------------------------------------------
    # Step 4: Make Predictions using Keras
    # -------------------------------------------------------------------------
    print("\n🎯 Step 4: Make Predictions using Keras")
    print("-" * 80)
    
    # Make predictions using the Keras model
    Y_pred_keras = model.predict(X_train, verbose=0)
    
    print(f"\n{'Hours':<10} {'z = wx+b':<15} {'Sigmoid(z)':<15} {'Prediction':<15}")
    print("-" * 55)
    
    for i, (x, y_pred) in enumerate(zip(X_train, Y_pred_keras)):
        z_value = w * x[0] + b
        prediction = "Pass (1)" if y_pred[0] >= 0.5 else "Fail (0)"
        print(f"{x[0]:<10.1f} {z_value:<15.2f} {y_pred[0]:<15.6f} {prediction:<15}")
    
    # -------------------------------------------------------------------------
    # Step 5: Manual Calculation using NumPy and Sigmoid
    # -------------------------------------------------------------------------
    print("\n🔍 Step 5: Manual Calculation using NumPy (Verification)")
    print("-" * 80)
    
    """
    Manual calculation of logistic regression:
    ------------------------------------------
    Step 1: Compute linear combination
        z = wx + b
    
    Step 2: Apply sigmoid activation
        g(z) = 1 / (1 + e^{-z})
    
    Combined:
        f_{w,b}(x) = 1 / (1 + e^{-(wx + b)})
    """
    
    # Manual computation
    W_manual = np.array([[w]])
    b_manual = np.array([b])
    
    # Step 1: Linear combination
    z_manual = np.dot(X_train, W_manual) + b_manual
    
    # Step 2: Sigmoid activation
    def manual_sigmoid(z):
        """
        Sigmoid function: g(z) = 1 / (1 + e^{-z})
        """
        return 1.0 / (1.0 + np.exp(-z))
    
    Y_pred_manual = manual_sigmoid(z_manual)
    
    print("Manual NumPy + Sigmoid predictions:")
    print(f"\n{'Hours':<10} {'z':<15} {'g(z)':<15} {'Formula':<40}")
    print("-" * 80)
    
    for i, (x, z, y_pred) in enumerate(zip(X_train, z_manual, Y_pred_manual)):
        formula = f"1/(1 + e^(-({w}*{x[0]:.1f} + {b})))"
        print(f"{x[0]:<10.1f} {z[0]:<15.2f} {y_pred[0]:<15.6f} {formula:<40}")
    
    # -------------------------------------------------------------------------
    # Step 6: Detailed Example for First Data Point
    # -------------------------------------------------------------------------
    print("\n🔬 Step 6: Detailed Calculation for X[0] = 0")
    print("-" * 80)
    
    x_example = X_train[0][0]
    
    print(f"\nInput: x = {x_example}")
    print(f"\nStep-by-step calculation:")
    print(f"  1. Linear combination:")
    print(f"     z = wx + b")
    print(f"     z = {w} * {x_example} + ({b})")
    print(f"     z = {w * x_example + b}")
    
    z_example = w * x_example + b
    
    print(f"\n  2. Sigmoid activation:")
    print(f"     g(z) = 1 / (1 + e^(-z))")
    print(f"     g({z_example}) = 1 / (1 + e^(-({z_example})))")
    print(f"     g({z_example}) = 1 / (1 + e^({-z_example}))")
    print(f"     g({z_example}) = 1 / (1 + {np.exp(-z_example):.6f})")
    print(f"     g({z_example}) = 1 / {1 + np.exp(-z_example):.6f}")
    
    y_example = manual_sigmoid(z_example)
    print(f"     g({z_example}) = {y_example:.6f}")
    
    print(f"\n  3. Keras prediction for X[0]:")
    print(f"     {Y_pred_keras[0][0]:.6f}")
    
    print(f"\n  4. Manual prediction for X[0]:")
    print(f"     {Y_pred_manual[0][0]:.6f}")
    
    match = "✓ Match" if np.allclose(Y_pred_keras[0], Y_pred_manual[0]) else "✗ Different"
    print(f"\n  Result: {match}")
    
    # -------------------------------------------------------------------------
    # Step 7: Compare Keras vs Manual Predictions
    # -------------------------------------------------------------------------
    print("\n📊 Step 7: Comparison - Keras vs Manual Predictions")
    print("-" * 80)
    
    print(f"\n{'Hours':<10} {'Keras':<15} {'Manual':<15} {'Difference':<15} {'Match?':<10}")
    print("-" * 65)
    
    for i in range(len(X_train)):
        keras_val = Y_pred_keras[i][0]
        manual_val = Y_pred_manual[i][0]
        diff = abs(keras_val - manual_val)
        match = "✓ Yes" if np.allclose(keras_val, manual_val) else "✗ No"
        print(f"{X_train[i][0]:<10.1f} {keras_val:<15.6f} {manual_val:<15.6f} "
              f"{diff:<15.2e} {match:<10}")
    
    # Verify all predictions match
    all_match = np.allclose(Y_pred_keras, Y_pred_manual)
    print(f"\n{'='*65}")
    print(f"✅ All predictions match: {all_match}")
    print(f"{'='*65}")
    
    # -------------------------------------------------------------------------
    # Step 8: Visualization
    # -------------------------------------------------------------------------
    print("\n📈 Step 8: Visualizing Logistic Regression")
    print("-" * 80)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: Sigmoid Function
    z_range = np.linspace(-10, 10, 200)
    sigmoid_values = manual_sigmoid(z_range)
    
    ax1.plot(z_range, sigmoid_values, 'b-', linewidth=2, label='g(z) = 1/(1+e^(-z))')
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Decision threshold')
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.set_xlabel('z = wx + b', fontsize=12)
    ax1.set_ylabel('g(z)', fontsize=12)
    ax1.set_title('Sigmoid Activation Function', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)
    
    # Subplot 2: Logistic Regression Predictions
    # Plot training data
    class_0 = Y_train.flatten() == 0
    class_1 = Y_train.flatten() == 1
    
    ax2.scatter(X_train[class_0], Y_train[class_0], color='red', s=100, 
                marker='o', alpha=0.6, label='Class 0 (Fail)', zorder=3, edgecolors='k')
    ax2.scatter(X_train[class_1], Y_train[class_1], color='blue', s=100, 
                marker='x', linewidth=2, label='Class 1 (Pass)', zorder=3)
    
    # Plot sigmoid curve
    x_range = np.linspace(-1, 6, 200).reshape(-1, 1)
    y_sigmoid = model.predict(x_range, verbose=0)
    ax2.plot(x_range, y_sigmoid, 'g-', linewidth=2, 
             label=f'f(x) = 1/(1+e^(-({w}x{b:+.1f})))', zorder=2)
    
    # Plot decision boundary
    decision_boundary = abs(b) / w
    ax2.axvline(x=decision_boundary, color='purple', linestyle='--', 
                linewidth=2, alpha=0.7, label=f'Decision boundary (x={decision_boundary:.2f})', zorder=1)
    ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, 
                label='Threshold (0.5)', zorder=1)
    
    ax2.set_xlabel('Hours Studied', fontsize=12)
    ax2.set_ylabel('Probability of Passing', fontsize=12)
    ax2.set_title('Logistic Regression - Single Neuron (Sigmoid Activation)', fontsize=14)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    plt.show()
    
    return model, Y_pred_keras, Y_pred_manual


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to run both sections of the neurons and layers demonstration.
    """
    print("\n" + "=" * 80)
    print(" " * 15 + "NEURONS AND LAYERS - TENSORFLOW/KERAS DEMONSTRATION")
    print(" " * 20 + "Coursera ML Specialization Implementation")
    print("=" * 80)
    
    print("\nThis script demonstrates:")
    print("  1. Linear Regression using a neuron WITHOUT activation (linear)")
    print("  2. Logistic Regression using a neuron WITH sigmoid activation")
    print("\nBoth examples compare Keras predictions with manual NumPy calculations.")
    
    input("\n⏎ Press Enter to start Section 1 (Linear Regression)...")
    
    # Run Section 1: Linear Regression
    linear_layer, keras_linear, manual_linear = section1_linear_regression()
    
    input("\n⏎ Press Enter to continue to Section 2 (Logistic Regression)...")
    
    # Run Section 2: Logistic Regression
    model, keras_logistic, manual_logistic = section2_logistic_regression()
    
    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" " * 30 + "FINAL SUMMARY")
    print("=" * 80)
    
    print("\n✅ Successfully demonstrated:")
    print("\n1. LINEAR REGRESSION (No Activation):")
    print(f"   - Formula: f(x) = wx + b")
    print(f"   - Keras predictions match manual calculations: ✓")
    print(f"   - Use case: Regression problems (continuous outputs)")
    
    print("\n2. LOGISTIC REGRESSION (Sigmoid Activation):")
    print(f"   - Formula: f(x) = 1/(1 + e^(-(wx + b)))")
    print(f"   - Keras predictions match manual calculations: ✓")
    print(f"   - Use case: Binary classification (probability outputs)")
    
    print("\n" + "=" * 80)
    print("Key Takeaways:")
    print("=" * 80)
    print("""
    1. A NEURON is a computational unit that:
       - Takes inputs (x)
       - Applies weights (w) and bias (b)
       - Optionally applies an activation function
       - Produces an output
    
    2. ACTIVATION FUNCTIONS determine the output behavior:
       - Linear: f(z) = z → Used for regression
       - Sigmoid: f(z) = 1/(1+e^(-z)) → Used for binary classification
       - Others: ReLU, tanh, softmax, etc.
    
    3. DENSE LAYER in Keras:
       - Fully connected layer
       - Each neuron connects to all inputs
       - Implements: output = activation(dot(input, weights) + bias)
    
    4. TensorFlow/Keras vs Manual Calculations:
       - Both produce identical results
       - Keras is optimized and handles backpropagation automatically
       - Manual calculations help understand the underlying mathematics
    """)
    
    print("=" * 80)
    print("✅ DEMONSTRATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run the demonstration
    main()
