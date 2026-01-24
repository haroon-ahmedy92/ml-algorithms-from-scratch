"""
Coffee Roasting Neural Network - TensorFlow/Keras Implementation
================================================================

This script demonstrates a binary classification neural network for coffee roasting quality
prediction, recreating the "Coffee Roasting" lab from the Coursera Deep Learning Specialization.

Problem Statement:
-----------------
Given coffee roasting parameters (Temperature and Duration), predict whether the roast
will produce good quality coffee (1) or poor quality coffee (0).

Author: ML Algorithms from Scratch Project
Date: January 24, 2026
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Normalization


# =============================================================================
# SECTION 1: DATA PREPARATION
# =============================================================================

def prepare_coffee_roasting_data():
    """
    Prepare synthetic coffee roasting dataset.
    
    Features:
    ---------
    X[:, 0]: Temperature (in Fahrenheit)
    X[:, 1]: Duration (in minutes)
    
    Target:
    -------
    Y: Binary label (1 = Good roast, 0 = Bad roast)
    
    Good coffee roasting typically occurs at:
    - Temperature: 200-215°F
    - Duration: 13-15 minutes
    """
    print("=" * 80)
    print("SECTION 1: DATA PREPARATION - COFFEE ROASTING DATASET")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Step 1: Create Initial Synthetic Dataset
    # -------------------------------------------------------------------------
    print("\n📊 Step 1: Creating Synthetic Coffee Roasting Dataset")
    print("-" * 80)
    
    # Initial small dataset (20 examples)
    np.random.seed(42)
    
    # Good roasts: Temperature ~200-215°F, Duration ~13-15 min
    X_good = np.array([
        [200, 13.9], [200, 14.0], [200, 14.5], [200, 15.0],
        [205, 13.5], [205, 14.0], [205, 14.5], [205, 15.0],
        [210, 13.5], [210, 14.0], [210, 14.5], [210, 15.0],
    ])
    Y_good = np.ones((len(X_good), 1))
    
    # Bad roasts: Too hot/cold or too long/short
    X_bad = np.array([
        [175, 13.0], [175, 16.0], [180, 12.5], [180, 17.0],
        [230, 13.0], [230, 16.0], [235, 12.5], [235, 17.0],
    ])
    Y_bad = np.zeros((len(X_bad), 1))
    
    # Combine datasets
    X_initial = np.vstack([X_good, X_bad])
    Y_initial = np.vstack([Y_good, Y_bad])
    
    print(f"Initial dataset created:")
    print(f"  - Number of examples: {len(X_initial)}")
    print(f"  - Features: Temperature (°F), Duration (min)")
    print(f"  - Good roasts: {np.sum(Y_initial == 1):.0f} examples")
    print(f"  - Bad roasts: {np.sum(Y_initial == 0):.0f} examples")
    print(f"\nSample data:")
    print(f"  First good roast: Temp={X_initial[0][0]:.1f}°F, Duration={X_initial[0][1]:.1f}min → Label={Y_initial[0][0]:.0f}")
    print(f"  First bad roast: Temp={X_initial[12][0]:.1f}°F, Duration={X_initial[12][1]:.1f}min → Label={Y_initial[12][0]:.0f}")
    
    # -------------------------------------------------------------------------
    # Step 2: Expand Dataset using np.tile
    # -------------------------------------------------------------------------
    print("\n📈 Step 2: Expanding Dataset for Training Efficiency")
    print("-" * 80)
    
    """
    Using np.tile to replicate the dataset:
    ---------------------------------------
    np.tile(array, reps) repeats the array 'reps' times along each axis.
    
    For a 2D array of shape (m, n):
        np.tile(array, (k, 1)) creates (k*m, n) by repeating rows k times
    
    This creates a larger training set for better gradient descent convergence.
    """
    
    # Calculate repetitions needed to reach ~200,000 examples
    target_size = 200000
    repetitions = target_size // len(X_initial)
    
    # Expand dataset
    X = np.tile(X_initial, (repetitions, 1))
    Y = np.tile(Y_initial, (repetitions, 1))
    
    print(f"Dataset expansion using np.tile:")
    print(f"  - Original size: {len(X_initial)} examples")
    print(f"  - Repetitions: {repetitions}")
    print(f"  - Final size: {len(X)} examples")
    print(f"  - Shape: X {X.shape}, Y {Y.shape}")
    
    # -------------------------------------------------------------------------
    # Step 3: Feature Normalization
    # -------------------------------------------------------------------------
    print("\n🔧 Step 3: Feature Normalization (Z-Score Normalization)")
    print("-" * 80)
    
    """
    Feature Normalization (Z-Score):
    --------------------------------
    For each feature j:
        x_norm[j] = (x[j] - μ[j]) / σ[j]
    
    Where:
        μ[j] = mean of feature j
        σ[j] = standard deviation of feature j
    
    LaTeX Formula:
        $$x_{norm}^{(i)}_j = \\frac{x^{(i)}_j - \\mu_j}{\\sigma_j}$$
    
    Benefits:
        - Ensures all features have similar scales
        - Improves gradient descent convergence
        - Prevents features with large values from dominating
    """
    
    # Create normalization layer
    norm_layer = Normalization(axis=-1)
    
    # Adapt the normalization layer to the data (computes mean and variance)
    norm_layer.adapt(X)
    
    # Get the computed statistics
    mean = norm_layer.mean.numpy().flatten()
    variance = norm_layer.variance.numpy().flatten()
    std = np.sqrt(variance)
    
    print("Normalization statistics computed:")
    print(f"  Temperature (Feature 0):")
    print(f"    - Mean (μ₀): {float(mean[0]):.2f}°F")
    print(f"    - Std Dev (σ₀): {float(std[0]):.2f}°F")
    print(f"  Duration (Feature 1):")
    print(f"    - Mean (μ₁): {float(mean[1]):.2f} min")
    print(f"    - Std Dev (σ₁): {float(std[1]):.2f} min")
    
    # Apply normalization
    X_normalized = norm_layer(X)
    
    print(f"\nNormalized data statistics:")
    print(f"  - Mean: {np.mean(X_normalized, axis=0)}")
    print(f"  - Std Dev: {np.std(X_normalized, axis=0)}")
    print(f"\nOriginal range:")
    print(f"  - Temperature: [{np.min(X[:, 0]):.1f}, {np.max(X[:, 0]):.1f}]°F")
    print(f"  - Duration: [{np.min(X[:, 1]):.1f}, {np.max(X[:, 1]):.1f}] min")
    print(f"\nNormalized range:")
    print(f"  - Temperature: [{np.min(X_normalized[:, 0]):.2f}, {np.max(X_normalized[:, 0]):.2f}]")
    print(f"  - Duration: [{np.min(X_normalized[:, 1]):.2f}, {np.max(X_normalized[:, 1]):.2f}]")
    
    return X, Y, X_normalized, norm_layer, X_initial, Y_initial


# =============================================================================
# SECTION 2: MODEL ARCHITECTURE
# =============================================================================

def build_coffee_roasting_model():
    """
    Build a neural network for coffee roasting quality prediction.
    
    Architecture:
    ------------
    Input Layer: 2 features (Temperature, Duration)
    Hidden Layer 1: 3 units, Sigmoid activation
    Output Layer: 1 unit, Sigmoid activation
    
    Mathematical Formulation:
    ------------------------
    Layer 1 (Hidden Layer):
        $$z^{[1]} = W^{[1]} \\cdot x + b^{[1]}$$
        $$a^{[1]} = g(z^{[1]})$$  where g is sigmoid
    
    Layer 2 (Output Layer):
        $$z^{[2]} = W^{[2]} \\cdot a^{[1]} + b^{[2]}$$
        $$a^{[2]} = g(z^{[2]})$$  where g is sigmoid
    
    Sigmoid Activation Function:
        $$g(z) = \\frac{1}{1 + e^{-z}}$$
    
    Vectorized Form for Layer l:
        $$a^{[l]} = g(W^{[l]} \\cdot a^{[l-1]} + b^{[l]})$$
    """
    print("\n" + "=" * 80)
    print("SECTION 2: MODEL ARCHITECTURE - NEURAL NETWORK DESIGN")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Step 1: Define Model Architecture
    # -------------------------------------------------------------------------
    print("\n🧠 Step 1: Defining Neural Network Architecture")
    print("-" * 80)
    
    print("Model architecture:")
    print("  Input Layer: 2 features (Temperature, Duration)")
    print("  Hidden Layer 1: 3 units with Sigmoid activation")
    print("  Output Layer: 1 unit with Sigmoid activation")
    
    # Build Sequential model
    model = Sequential([
        Input(shape=(2,)),  # 2 input features
        Dense(units=3, activation='sigmoid', name='layer1'),  # Hidden layer
        Dense(units=1, activation='sigmoid', name='layer2')   # Output layer
    ])
    
    print("\n✓ Model created successfully")
    
    # -------------------------------------------------------------------------
    # Step 2: Display Model Summary
    # -------------------------------------------------------------------------
    print("\n📋 Step 2: Model Summary and Parameter Count")
    print("-" * 80)
    
    model.summary()
    
    # -------------------------------------------------------------------------
    # Step 3: Explain Parameter Counts
    # -------------------------------------------------------------------------
    print("\n🔍 Step 3: Parameter Count Explanation")
    print("-" * 80)
    
    print("\nLayer 1 (Hidden Layer): 3 units, 2 inputs")
    print("  - Weight matrix W^[1]: shape (2, 3)")
    print("    Each of 3 units has 2 weights (one per input feature)")
    print("    Total weights: 2 × 3 = 6")
    print("  - Bias vector b^[1]: shape (3,)")
    print("    Each of 3 units has 1 bias")
    print("    Total biases: 3")
    print("  - Total parameters for Layer 1: 6 + 3 = 9")
    
    print("\nLayer 2 (Output Layer): 1 unit, 3 inputs (from Layer 1)")
    print("  - Weight matrix W^[2]: shape (3, 1)")
    print("    The 1 output unit has 3 weights (one per hidden unit)")
    print("    Total weights: 3 × 1 = 3")
    print("  - Bias vector b^[2]: shape (1,)")
    print("    The 1 output unit has 1 bias")
    print("    Total biases: 1")
    print("  - Total parameters for Layer 2: 3 + 1 = 4")
    
    print("\nTotal trainable parameters: 9 + 4 = 13")
    
    print("\n" + "=" * 80)
    print("Mathematical Formulation:")
    print("=" * 80)
    print("""
For an input x = [Temperature, Duration]:

Layer 1 Computation:
    z^[1] = W^[1] · x + b^[1]
    a^[1] = sigmoid(z^[1])
    
    Where:
        W^[1] is (3 × 2) matrix
        x is (2 × 1) vector
        b^[1] is (3 × 1) vector
        a^[1] is (3 × 1) vector (output of hidden layer)

Layer 2 Computation:
    z^[2] = W^[2] · a^[1] + b^[2]
    a^[2] = sigmoid(z^[2])
    
    Where:
        W^[2] is (1 × 3) matrix
        a^[1] is (3 × 1) vector
        b^[2] is (1 × 1) scalar
        a^[2] is (1 × 1) scalar (final prediction probability)

Sigmoid Function:
    g(z) = 1 / (1 + e^(-z))
    """)
    
    return model


# =============================================================================
# SECTION 3: MODEL TRAINING
# =============================================================================

def train_coffee_roasting_model(model, X_train, Y_train):
    """
    Train the neural network using Binary Cross-Entropy loss and Adam optimizer.
    
    Binary Cross-Entropy Loss:
    --------------------------
    For a single example (x, y):
        $$L(\\hat{y}, y) = -[y \\log(\\hat{y}) + (1-y) \\log(1-\\hat{y})]$$
    
    For m training examples:
        $$J(W,b) = -\\frac{1}{m} \\sum_{i=1}^{m} [y^{(i)} \\log(\\hat{y}^{(i)}) + (1-y^{(i)}) \\log(1-\\hat{y}^{(i)})]$$
    
    Where:
        y = true label (0 or 1)
        ŷ = predicted probability (output of network)
        m = number of training examples
    
    Adam Optimizer:
    --------------
    Adaptive Moment Estimation - combines benefits of:
        - Momentum: Uses exponentially weighted averages of gradients
        - RMSprop: Adapts learning rate per parameter
    
    Learning rate: 0.01
    """
    print("\n" + "=" * 80)
    print("SECTION 3: MODEL TRAINING")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Step 1: Compile Model
    # -------------------------------------------------------------------------
    print("\n⚙️  Step 1: Compiling Model")
    print("-" * 80)
    
    print("Loss function: Binary Cross-Entropy")
    print("  Formula: J = -1/m × Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]")
    print("\nOptimizer: Adam (Adaptive Moment Estimation)")
    print("  Learning rate: 0.01")
    print("\nMetrics: Binary Accuracy")
    
    model.compile(
        loss=BinaryCrossentropy(),
        optimizer=Adam(learning_rate=0.01),
        metrics=['binary_accuracy']
    )
    
    print("\n✓ Model compiled successfully")
    
    # -------------------------------------------------------------------------
    # Step 2: Train Model
    # -------------------------------------------------------------------------
    print("\n🎓 Step 2: Training Neural Network")
    print("-" * 80)
    
    print(f"Training dataset size: {len(X_train)} examples")
    print(f"Number of epochs: 10")
    print("\nTraining in progress...\n")
    
    # Train the model
    history = model.fit(
        X_train,
        Y_train,
        epochs=10,
        batch_size=32,
        verbose=1,
        validation_split=0.1  # Use 10% for validation
    )
    
    print("\n✓ Training completed!")
    
    # -------------------------------------------------------------------------
    # Step 3: Training Results
    # -------------------------------------------------------------------------
    print("\n📊 Step 3: Training Results")
    print("-" * 80)
    
    final_loss = history.history['loss'][-1]
    final_accuracy = history.history['binary_accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_val_accuracy = history.history['val_binary_accuracy'][-1]
    
    print(f"\nFinal training metrics:")
    print(f"  - Loss: {final_loss:.6f}")
    print(f"  - Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"\nFinal validation metrics:")
    print(f"  - Loss: {final_val_loss:.6f}")
    print(f"  - Accuracy: {final_val_accuracy:.4f} ({final_val_accuracy*100:.2f}%)")
    
    return history


# =============================================================================
# SECTION 4: PREDICTION AND DECISION MAKING
# =============================================================================

def make_predictions(model, norm_layer):
    """
    Make predictions on test data and apply decision threshold.
    
    Decision Rule:
    -------------
    For a predicted probability ŷ:
        - If ŷ >= 0.5: Predict class 1 (Good roast)
        - If ŷ < 0.5: Predict class 0 (Bad roast)
    
    Threshold Selection:
        $$\\text{decision} = \\begin{cases} 
            1 & \\text{if } \\hat{y} \\geq 0.5 \\\\
            0 & \\text{if } \\hat{y} < 0.5
        \\end{cases}$$
    """
    print("\n" + "=" * 80)
    print("SECTION 4: PREDICTION AND DECISION MAKING")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Step 1: Define Test Cases
    # -------------------------------------------------------------------------
    print("\n🧪 Step 1: Define Test Cases")
    print("-" * 80)
    
    # Test cases
    X_test = np.array([
        [200, 13.9],  # Good roast: optimal temperature and duration
        [200, 17.0],  # Bad roast: good temperature but too long duration
    ])
    
    print("Test cases:")
    for i, x in enumerate(X_test):
        print(f"  Test {i+1}: Temperature = {x[0]:.1f}°F, Duration = {x[1]:.1f} min")
    
    # -------------------------------------------------------------------------
    # Step 2: Normalize Test Data
    # -------------------------------------------------------------------------
    print("\n🔧 Step 2: Normalize Test Data")
    print("-" * 80)
    
    # Apply the same normalization as training data
    X_test_normalized = norm_layer(X_test)
    
    print("Normalized test data:")
    for i, (x_orig, x_norm) in enumerate(zip(X_test, X_test_normalized)):
        print(f"  Test {i+1}:")
        print(f"    Original: [{x_orig[0]:.1f}, {x_orig[1]:.1f}]")
        print(f"    Normalized: [{x_norm[0]:.4f}, {x_norm[1]:.4f}]")
    
    # -------------------------------------------------------------------------
    # Step 3: Make Predictions
    # -------------------------------------------------------------------------
    print("\n🎯 Step 3: Make Predictions (Probabilities)")
    print("-" * 80)
    
    # Get probability predictions
    predictions_prob = model.predict(X_test_normalized, verbose=0)
    
    print("\nPredicted probabilities:")
    for i, (x, prob) in enumerate(zip(X_test, predictions_prob)):
        print(f"  Test {i+1}: Temp={x[0]:.1f}°F, Duration={x[1]:.1f}min")
        print(f"    → P(Good roast) = {prob[0]:.6f}")
    
    # -------------------------------------------------------------------------
    # Step 4: Apply Decision Threshold
    # -------------------------------------------------------------------------
    print("\n📏 Step 4: Apply Decision Threshold (0.5)")
    print("-" * 80)
    
    print("Decision rule:")
    print("  - If P(Good roast) >= 0.5: Predict 1 (Good roast)")
    print("  - If P(Good roast) < 0.5: Predict 0 (Bad roast)")
    
    # Apply threshold
    threshold = 0.5
    predictions_binary = (predictions_prob >= threshold).astype(int)
    
    print("\nFinal predictions:")
    print(f"\n{'Test':<8} {'Temp (°F)':<12} {'Duration (min)':<16} {'Probability':<15} {'Decision':<12} {'Prediction':<15}")
    print("-" * 80)
    
    for i, (x, prob, pred) in enumerate(zip(X_test, predictions_prob, predictions_binary)):
        decision = "Good" if pred[0] == 1 else "Bad"
        color = "✓" if pred[0] == 1 else "✗"
        print(f"{i+1:<8} {x[0]:<12.1f} {x[1]:<16.1f} {prob[0]:<15.6f} {pred[0]:<12} {color} {decision} roast")
    
    # -------------------------------------------------------------------------
    # Step 5: Interpretation
    # -------------------------------------------------------------------------
    print("\n💡 Step 5: Interpretation")
    print("-" * 80)
    
    print("\nResults interpretation:")
    if predictions_binary[0][0] == 1:
        print(f"  Test 1 (200°F, 13.9min): GOOD roast - Within optimal range")
    else:
        print(f"  Test 1 (200°F, 13.9min): BAD roast - Outside optimal range")
    
    if predictions_binary[1][0] == 1:
        print(f"  Test 2 (200°F, 17.0min): GOOD roast - Within optimal range")
    else:
        print(f"  Test 2 (200°F, 17.0min): BAD roast - Duration too long")
    
    print("\nThe neural network has learned the relationship between:")
    print("  - Temperature and Duration → Coffee roasting quality")
    print("  - Optimal range: ~200-215°F, ~13-15 minutes")
    
    return predictions_prob, predictions_binary


# =============================================================================
# SECTION 5: VISUALIZATION
# =============================================================================

def visualize_results(X_initial, Y_initial, model, norm_layer, history):
    """
    Visualize the training process and decision boundary.
    """
    print("\n" + "=" * 80)
    print("SECTION 5: VISUALIZATION")
    print("=" * 80)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # -------------------------------------------------------------------------
    # Plot 1: Training History
    # -------------------------------------------------------------------------
    ax = axes[0]
    epochs = range(1, len(history.history['loss']) + 1)
    
    ax.plot(epochs, history.history['loss'], 'b-', linewidth=2, label='Training Loss')
    ax.plot(epochs, history.history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Binary Cross-Entropy Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 2: Accuracy History
    # -------------------------------------------------------------------------
    ax = axes[1]
    ax.plot(epochs, history.history['binary_accuracy'], 'b-', linewidth=2, label='Training Accuracy')
    ax.plot(epochs, history.history['val_binary_accuracy'], 'r-', linewidth=2, label='Validation Accuracy')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # -------------------------------------------------------------------------
    # Plot 3: Decision Boundary (Original Data)
    # -------------------------------------------------------------------------
    ax = axes[2]
    
    # Plot training data
    good_roasts = Y_initial.flatten() == 1
    bad_roasts = Y_initial.flatten() == 0
    
    ax.scatter(X_initial[good_roasts, 0], X_initial[good_roasts, 1],
               c='green', marker='o', s=100, alpha=0.6, edgecolors='k',
               label='Good Roast (1)', zorder=3)
    ax.scatter(X_initial[bad_roasts, 0], X_initial[bad_roasts, 1],
               c='red', marker='x', s=100, linewidths=2,
               label='Bad Roast (0)', zorder=3)
    
    # Create mesh for decision boundary
    temp_range = np.linspace(170, 240, 200)
    duration_range = np.linspace(12, 18, 200)
    temp_mesh, duration_mesh = np.meshgrid(temp_range, duration_range)
    
    # Prepare data for prediction
    mesh_data = np.c_[temp_mesh.ravel(), duration_mesh.ravel()]
    mesh_normalized = norm_layer(mesh_data)
    
    # Get predictions
    predictions = model.predict(mesh_normalized, verbose=0)
    predictions = predictions.reshape(temp_mesh.shape)
    
    # Plot decision boundary
    contour = ax.contourf(temp_mesh, duration_mesh, predictions,
                          levels=[0, 0.5, 1], alpha=0.3,
                          colors=['red', 'green'], zorder=1)
    ax.contour(temp_mesh, duration_mesh, predictions,
               levels=[0.5], colors='black', linewidths=2, zorder=2)
    
    ax.set_xlabel('Temperature (°F)', fontsize=12)
    ax.set_ylabel('Duration (minutes)', fontsize=12)
    ax.set_title('Decision Boundary - Coffee Roasting', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n✓ Visualizations generated successfully")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to run the complete Coffee Roasting Neural Network demonstration.
    """
    print("\n" + "=" * 80)
    print(" " * 15 + "COFFEE ROASTING NEURAL NETWORK - COMPLETE DEMONSTRATION")
    print(" " * 20 + "Coursera Deep Learning Specialization")
    print("=" * 80)
    
    print("\n🎯 Objective:")
    print("Build a neural network to predict coffee roasting quality based on")
    print("temperature and duration parameters.")
    
    print("\n📋 Overview:")
    print("  1. Prepare and normalize coffee roasting dataset")
    print("  2. Build a 2-layer neural network (3 hidden units, sigmoid activation)")
    print("  3. Train using Binary Cross-Entropy loss and Adam optimizer")
    print("  4. Make predictions and apply decision threshold")
    print("  5. Visualize results and decision boundary")
    
    input("\n⏎ Press Enter to start...")
    
    # Section 1: Data Preparation
    X, Y, X_normalized, norm_layer, X_initial, Y_initial = prepare_coffee_roasting_data()
    
    input("\n⏎ Press Enter to continue to Model Architecture...")
    
    # Section 2: Model Architecture
    model = build_coffee_roasting_model()
    
    input("\n⏎ Press Enter to continue to Training...")
    
    # Section 3: Training
    history = train_coffee_roasting_model(model, X_normalized, Y)
    
    input("\n⏎ Press Enter to continue to Predictions...")
    
    # Section 4: Predictions
    predictions_prob, predictions_binary = make_predictions(model, norm_layer)
    
    input("\n⏎ Press Enter to view visualizations...")
    
    # Section 5: Visualization
    visualize_results(X_initial, Y_initial, model, norm_layer, history)
    
    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" " * 30 + "FINAL SUMMARY")
    print("=" * 80)
    
    print("\n✅ Successfully demonstrated:")
    print("\n1. DATA PREPARATION:")
    print("   ✓ Created synthetic coffee roasting dataset")
    print("   ✓ Expanded dataset to 200,000 examples using np.tile")
    print("   ✓ Applied Z-score normalization using Keras Normalization layer")
    
    print("\n2. MODEL ARCHITECTURE:")
    print("   ✓ Input Layer: 2 features (Temperature, Duration)")
    print("   ✓ Hidden Layer: 3 units with Sigmoid activation (9 parameters)")
    print("   ✓ Output Layer: 1 unit with Sigmoid activation (4 parameters)")
    print("   ✓ Total: 13 trainable parameters")
    
    print("\n3. TRAINING:")
    print("   ✓ Loss: Binary Cross-Entropy")
    print("   ✓ Optimizer: Adam (learning rate = 0.01)")
    print("   ✓ Trained for 10 epochs")
    print(f"   ✓ Final accuracy: {history.history['binary_accuracy'][-1]*100:.2f}%")
    
    print("\n4. PREDICTIONS:")
    print("   ✓ Applied normalization to test data")
    print("   ✓ Generated probability predictions")
    print("   ✓ Applied 0.5 threshold for binary classification")
    
    print("\n" + "=" * 80)
    print("Key Concepts Demonstrated:")
    print("=" * 80)
    print("""
1. NORMALIZATION: Scales features to have mean=0, std=1
   Formula: x_norm = (x - μ) / σ

2. NEURAL NETWORK FORWARD PROPAGATION:
   Layer l: a^[l] = g(W^[l] · a^[l-1] + b^[l])

3. SIGMOID ACTIVATION: Maps values to (0, 1) range
   Formula: g(z) = 1 / (1 + e^(-z))

4. BINARY CROSS-ENTROPY LOSS: Measures prediction error
   Formula: J = -1/m × Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]

5. DECISION THRESHOLD: Converts probabilities to classes
   Rule: predict 1 if ŷ >= 0.5, else 0
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
