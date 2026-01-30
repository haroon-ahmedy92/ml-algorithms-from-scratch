"""
ML Model Diagnostics Pipeline - Bias, Variance, and Overfitting Analysis
========================================================================

This script implements a comprehensive diagnostic pipeline for evaluating and improving
machine learning models. It covers bias-variance tradeoffs, regularization techniques,
and neural network complexity analysis.

Educational Objective:
----------------------
Understanding how to diagnose and improve ML models by:
1. Identifying high bias vs high variance scenarios
2. Finding optimal model complexity 
3. Using regularization to combat overfitting
4. Comparing simple vs complex neural networks

Mathematical Foundation:
------------------------
Mean Squared Error (MSE):
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

Ridge Regularization:
$$J_{regularized}(\theta) = J(\theta) + \lambda \sum_{j=1}^{n} \theta_j^2$$

Classification Error:
$$\text{Error} = \frac{1}{m} \sum_{i=1}^{m} \mathbf{1}[h(x^{(i)}) \neq y^{(i)}]$$

Key Concepts:
-------------
- **High Bias (Underfitting)**: Model is too simple, poor performance on both training and validation
- **High Variance (Overfitting)**: Model is too complex, good training performance but poor validation
- **Optimal Complexity**: Sweet spot that minimizes validation error

Author: ML Algorithms from Scratch Project
Date: January 30, 2026
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.datasets import make_regression, make_classification
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 1: DATA GENERATION AND SPLITTING
# =============================================================================

def generate_regression_data(n_samples=1000, noise=0.1, random_state=42):
    """
    Generates synthetic regression data for bias-variance analysis.
    
    Returns:
        X (ndarray): Shape (n_samples, 1) - Input features
        y (ndarray): Shape (n_samples,) - Target values
    """
    print("=" * 70)
    print("ML Model Diagnostics Pipeline")
    print("=" * 70)
    
    # Generate non-linear data
    np.random.seed(random_state)
    X = np.linspace(0, 1, n_samples).reshape(-1, 1)
    y = 1.5 * X.flatten() + 0.5 * np.sin(15 * X.flatten()) + noise * np.random.randn(n_samples)
    
    print(f"\n✓ Generated Regression Dataset:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Features: Non-linear function with noise")
    
    return X, y

def generate_classification_data(n_samples=1000, n_features=20, n_classes=3, random_state=42):
    """
    Generates synthetic classification data for neural network analysis.
    
    Returns:
        X (ndarray): Shape (n_samples, n_features) - Input features
        y (ndarray): Shape (n_samples,) - Class labels
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        n_classes=n_classes,
        random_state=random_state
    )
    
    print(f"\n✓ Generated Classification Dataset:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Classes: {n_classes}")
    print(f"  Class distribution: {np.bincount(y)}")
    
    return X, y

def split_data(X, y, train_size=0.6, cv_size=0.2, test_size=0.2, random_state=42):
    """
    Splits data into Training (60%), Cross-Validation (20%), and Test (20%) sets.
    
    Data Splitting Strategy:
    ------------------------
    1. Training Set (60%): Used to fit model parameters
    2. Cross-Validation Set (20%): Used for model selection and hyperparameter tuning
    3. Test Set (20%): Used for final unbiased evaluation
    
    This 3-way split prevents data leakage and provides honest performance estimates.
    
    Parameters:
        X, y: Input features and targets
        train_size, cv_size, test_size: Split proportions (must sum to 1.0)
        
    Returns:
        X_train, X_cv, X_test, y_train, y_cv, y_test: Split datasets
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: training and cross-validation from remaining data
    relative_cv_size = cv_size / (train_size + cv_size)
    X_train, X_cv, y_train, y_cv = train_test_split(
        X_temp, y_temp, test_size=relative_cv_size, random_state=random_state
    )
    
    print(f"\n✓ Data Split Complete:")
    print(f"  Training Set:   {X_train.shape[0]} samples ({train_size*100:.0f}%)")
    print(f"  CV Set:         {X_cv.shape[0]} samples ({cv_size*100:.0f}%)")
    print(f"  Test Set:       {X_test.shape[0]} samples ({test_size*100:.0f}%)")
    
    return X_train, X_cv, X_test, y_train, y_cv, y_test

# =============================================================================
# SECTION 2: ERROR FUNCTIONS
# =============================================================================

def eval_mse(y, yhat):
    """
    Calculates Mean Squared Error using the standard ML formula.
    
    Mathematical Definition:
    ------------------------
    $$J = \frac{1}{2m} \sum_{i=1}^{m} (\\hat{y}^{(i)} - y^{(i)})^2$$
    
    The factor of 1/2 is often used in ML for mathematical convenience
    (simplifies derivatives during gradient descent).
    
    Parameters:
        y (ndarray): True values
        yhat (ndarray): Predicted values
        
    Returns:
        float: Mean squared error
    """
    m = len(y)
    mse = (1 / (2 * m)) * np.sum((yhat - y) ** 2)
    return mse

def eval_cat_err(y, yhat):
    """
    Calculates classification error as fraction of misclassified examples.
    
    Mathematical Definition:
    ------------------------
    $$\\text{Error} = \\frac{1}{m} \\sum_{i=1}^{m} \\mathbf{1}[h(x^{(i)}) \\neq y^{(i)}]$$
    
    Where $\\mathbf{1}[\\cdot]$ is the indicator function (1 if condition is true, 0 otherwise).
    
    Parameters:
        y (ndarray): True class labels
        yhat (ndarray): Predicted class labels
        
    Returns:
        float: Classification error rate (0 to 1)
    """
    m = len(y)
    error = (1 / m) * np.sum(y != yhat)
    return error

# =============================================================================
# SECTION 3: POLYNOMIAL REGRESSION DIAGNOSTICS
# =============================================================================

def demonstrate_overfitting(X_train, X_test, y_train, y_test, degree=10):
    """
    Demonstrates overfitting using high-degree polynomial regression.
    
    Overfitting Symptoms:
    ---------------------
    - Very low training error (model memorizes training data)
    - High test error (poor generalization to unseen data)
    - Large gap between training and test performance
    
    Parameters:
        X_train, X_test, y_train, y_test: Training and test sets
        degree: Polynomial degree (high values cause overfitting)
        
    Returns:
        train_error, test_error: MSE on training and test sets
    """
    print(f"\n" + "=" * 70)
    print(f"Demonstrating Overfitting with Polynomial Degree {degree}")
    print("=" * 70)
    
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    
    # Fit linear regression on polynomial features
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    # Calculate errors
    train_error = eval_mse(y_train, y_train_pred)
    test_error = eval_mse(y_test, y_test_pred)
    
    print(f"\nPolynomial Degree {degree} Results:")
    print(f"  Training Error (MSE): {train_error:.6f}")
    print(f"  Test Error (MSE):     {test_error:.6f}")
    print(f"  Error Ratio (Test/Train): {test_error/train_error:.2f}")
    
    if test_error > 2 * train_error:
        print("  🚨 OVERFITTING DETECTED: Test error >> Training error")
    else:
        print("  ✅ Model appears well-balanced")
    
    return train_error, test_error

def bias_variance_analysis(X_train, X_cv, y_train, y_cv, max_degree=9):
    """
    Performs bias-variance analysis by varying polynomial complexity.
    
    Analysis Strategy:
    ------------------
    1. Fit polynomials of degrees 1 to max_degree
    2. Calculate training and CV errors for each degree
    3. Identify patterns:
       - High bias: Both errors high and close together
       - High variance: Large gap between training and CV errors
       - Optimal: Minimum CV error
    
    Parameters:
        X_train, X_cv, y_train, y_cv: Training and cross-validation sets
        max_degree: Maximum polynomial degree to test
        
    Returns:
        degrees, train_errors, cv_errors: Arrays for plotting learning curves
    """
    print(f"\n" + "=" * 70)
    print(f"Bias-Variance Analysis (Polynomial Degrees 1-{max_degree})")
    print("=" * 70)
    
    degrees = range(1, max_degree + 1)
    train_errors = []
    cv_errors = []
    
    for degree in degrees:
        # Create polynomial features
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly_features.fit_transform(X_train)
        X_cv_poly = poly_features.transform(X_cv)
        
        # Fit model
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train_poly)
        y_cv_pred = model.predict(X_cv_poly)
        
        # Calculate errors
        train_error = eval_mse(y_train, y_train_pred)
        cv_error = eval_mse(y_cv, y_cv_pred)
        
        train_errors.append(train_error)
        cv_errors.append(cv_error)
        
        print(f"  Degree {degree}: Train={train_error:.6f}, CV={cv_error:.6f}")
    
    # Find optimal degree
    optimal_degree = degrees[np.argmin(cv_errors)]
    min_cv_error = min(cv_errors)
    
    print(f"\n✓ Optimal Polynomial Degree: {optimal_degree}")
    print(f"  Minimum CV Error: {min_cv_error:.6f}")
    
    return list(degrees), train_errors, cv_errors

# =============================================================================
# SECTION 4: REGULARIZATION TUNING
# =============================================================================

def regularization_analysis(X_train, X_cv, y_train, y_cv, degree=10):
    """
    Analyzes the effect of L2 regularization (Ridge) on overfitting.
    
    Regularization Strategy:
    ------------------------
    Using a high-degree polynomial (prone to overfitting), we vary λ to find
    the value that best balances bias and variance.
    
    λ = 0:    No regularization (potential overfitting)
    λ small:  Light regularization (reduced overfitting)
    λ large:  Heavy regularization (potential underfitting)
    
    Parameters:
        X_train, X_cv, y_train, y_cv: Training and cross-validation sets
        degree: Fixed polynomial degree for regularization study
        
    Returns:
        lambdas, train_errors, cv_errors: Arrays for plotting regularization curves
    """
    print(f"\n" + "=" * 70)
    print(f"Regularization Analysis (Ridge Regression, Degree {degree})")
    print("=" * 70)
    
    # Range of lambda values (logarithmic scale)
    lambdas = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    train_errors = []
    cv_errors = []
    
    # Create polynomial features once
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train)
    X_cv_poly = poly_features.transform(X_cv)
    
    # Standardize features for regularization
    scaler = StandardScaler()
    X_train_poly = scaler.fit_transform(X_train_poly)
    X_cv_poly = scaler.transform(X_cv_poly)
    
    for lambda_val in lambdas:
        # Fit Ridge regression
        if lambda_val == 0.0:
            # Use LinearRegression when no regularization
            model = LinearRegression()
        else:
            model = Ridge(alpha=lambda_val)
        
        model.fit(X_train_poly, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train_poly)
        y_cv_pred = model.predict(X_cv_poly)
        
        # Calculate errors
        train_error = eval_mse(y_train, y_train_pred)
        cv_error = eval_mse(y_cv, y_cv_pred)
        
        train_errors.append(train_error)
        cv_errors.append(cv_error)
        
        print(f"  λ={lambda_val:8.0e}: Train={train_error:.6f}, CV={cv_error:.6f}")
    
    # Find optimal lambda
    optimal_idx = np.argmin(cv_errors)
    optimal_lambda = lambdas[optimal_idx]
    min_cv_error = cv_errors[optimal_idx]
    
    print(f"\n✓ Optimal Regularization: λ = {optimal_lambda}")
    print(f"  Minimum CV Error: {min_cv_error:.6f}")
    
    return lambdas, train_errors, cv_errors

# =============================================================================
# SECTION 5: NEURAL NETWORK DIAGNOSTICS
# =============================================================================

def build_simple_model(input_dim, num_classes):
    """
    Builds a simple neural network with limited capacity.
    
    Architecture:
    -------------
    Input → Dense(6, ReLU) → Dense(num_classes, Linear)
    
    Simple models are prone to:
    - High bias (underfitting)
    - Good generalization (if not too simple)
    """
    model = Sequential([
        Dense(6, activation='relu', input_shape=(input_dim,), name='hidden'),
        Dense(num_classes, activation='linear', name='output')
    ], name='simple_model')
    
    model.compile(
        loss=SparseCategoricalCrossentropy(from_logits=True),
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    return model

def build_complex_model(input_dim, num_classes, l2_reg=0.01):
    """
    Builds a complex neural network with L2 regularization.
    
    Architecture:
    -------------
    Input → Dense(120, ReLU, L2) → Dense(40, ReLU, L2) → Dense(num_classes, Linear)
    
    Complex models are prone to:
    - High variance (overfitting)
    - Better representation capability
    - Need regularization to control overfitting
    
    L2 Regularization:
    ------------------
    Adds penalty term: λ * sum(w²) to the loss function
    Forces weights to be small, reducing model complexity
    """
    model = Sequential([
        Dense(120, activation='relu', 
              kernel_regularizer=l2(l2_reg),
              input_shape=(input_dim,), name='hidden1'),
        Dense(40, activation='relu',
              kernel_regularizer=l2(l2_reg), name='hidden2'),
        Dense(num_classes, activation='linear', name='output')
    ], name='complex_model')
    
    model.compile(
        loss=SparseCategoricalCrossentropy(from_logits=True),
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    return model

def compare_neural_networks(X_train, X_cv, y_train, y_cv, epochs=50):
    """
    Compares simple vs complex neural network architectures.
    
    Comparison Objectives:
    ----------------------
    1. Simple model: Check for high bias (underfitting)
    2. Complex model: Check for high variance (overfitting)
    3. Regularized complex model: Check if regularization helps
    
    Parameters:
        X_train, X_cv, y_train, y_cv: Training and validation sets
        epochs: Number of training epochs
        
    Returns:
        Dictionary with model histories and performance metrics
    """
    print(f"\n" + "=" * 70)
    print("Neural Network Architecture Comparison")
    print("=" * 70)
    
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    print(f"\nDataset Info:")
    print(f"  Input Dimension: {input_dim}")
    print(f"  Number of Classes: {num_classes}")
    print(f"  Training Samples: {len(X_train)}")
    print(f"  Validation Samples: {len(X_cv)}")
    
    # Build models
    simple_model = build_simple_model(input_dim, num_classes)
    complex_model = build_complex_model(input_dim, num_classes, l2_reg=0.01)
    
    print(f"\nModel Architectures:")
    print(f"  Simple:  {simple_model.count_params()} parameters")
    print(f"  Complex: {complex_model.count_params()} parameters")
    
    # Train simple model
    print(f"\n📊 Training Simple Model ({epochs} epochs)...")
    history_simple = simple_model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_cv, y_cv),
        verbose=1  # Show training progress
    )
    
    # Train complex model
    print(f"\n📊 Training Complex Model ({epochs} epochs)...")
    history_complex = complex_model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_cv, y_cv),
        verbose=1  # Show training progress
    )
    
    # Evaluate final performance
    simple_train_acc = history_simple.history['accuracy'][-1]
    simple_val_acc = history_simple.history['val_accuracy'][-1]
    complex_train_acc = history_complex.history['accuracy'][-1]
    complex_val_acc = history_complex.history['val_accuracy'][-1]
    
    print(f"\n✓ Final Performance:")
    print(f"  Simple Model:")
    print(f"    Training Accuracy:   {simple_train_acc*100:.2f}%")
    print(f"    Validation Accuracy: {simple_val_acc*100:.2f}%")
    print(f"    Gap: {abs(simple_train_acc - simple_val_acc)*100:.2f}%")
    
    print(f"  Complex Model (with L2 regularization):")
    print(f"    Training Accuracy:   {complex_train_acc*100:.2f}%")
    print(f"    Validation Accuracy: {complex_val_acc*100:.2f}%")
    print(f"    Gap: {abs(complex_train_acc - complex_val_acc)*100:.2f}%")
    
    # Diagnose models
    print(f"\n🔍 Model Diagnosis:")
    if simple_train_acc < 0.7 and simple_val_acc < 0.7:
        print("  Simple Model: HIGH BIAS (underfitting) - consider more complexity")
    elif abs(simple_train_acc - simple_val_acc) > 0.1:
        print("  Simple Model: HIGH VARIANCE (overfitting) - consider regularization")
    else:
        print("  Simple Model: Well-balanced")
    
    if abs(complex_train_acc - complex_val_acc) > 0.1:
        print("  Complex Model: HIGH VARIANCE (overfitting) - regularization helping but may need more")
    elif complex_train_acc < 0.7 and complex_val_acc < 0.7:
        print("  Complex Model: HIGH BIAS (underfitting) - reduce regularization")
    else:
        print("  Complex Model: Well-balanced with regularization")
    
    return {
        'simple_model': simple_model,
        'complex_model': complex_model,
        'history_simple': history_simple,
        'history_complex': history_complex
    }

# =============================================================================
# SECTION 6: VISUALIZATION
# =============================================================================

def plot_learning_curves(degrees, train_errors, cv_errors, title="Bias-Variance Analysis"):
    """
    Plots learning curves to identify bias and variance issues.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, train_errors, 'o-', linewidth=2, label='Training Error', color='blue')
    plt.plot(cv_errors, 'o-', linewidth=2, label='Cross-Validation Error', color='red')
    plt.xlabel('Polynomial Degree', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    optimal_degree = degrees[np.argmin(cv_errors)]
    plt.axvline(x=optimal_degree, color='green', linestyle='--', alpha=0.7)
    plt.text(optimal_degree + 0.1, max(cv_errors) * 0.8, 
             f'Optimal Degree: {optimal_degree}', fontsize=10, color='green')
    
    plt.tight_layout()
    plt.show()

def plot_regularization_curves(lambdas, train_errors, cv_errors):
    """
    Plots regularization curves to find optimal lambda.
    """
    plt.figure(figsize=(10, 6))
    plt.semilogx(lambdas[1:], train_errors[1:], 'o-', linewidth=2, label='Training Error', color='blue')
    plt.semilogx(lambdas[1:], cv_errors[1:], 'o-', linewidth=2, label='Cross-Validation Error', color='red')
    plt.xlabel('Regularization Parameter (λ)', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.title('Regularization Analysis - Finding Optimal λ', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add optimal lambda annotation
    optimal_idx = np.argmin(cv_errors)
    optimal_lambda = lambdas[optimal_idx]
    plt.axvline(x=optimal_lambda, color='green', linestyle='--', alpha=0.7)
    plt.text(optimal_lambda * 2, max(cv_errors) * 0.8, 
             f'Optimal λ: {optimal_lambda:.0e}', fontsize=10, color='green')
    
    plt.tight_layout()
    plt.show()

def plot_neural_network_comparison(results):
    """
    Plots training histories for neural network comparison.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Training and validation loss
    ax1.plot(results['history_simple'].history['loss'], label='Simple - Training', linewidth=2)
    ax1.plot(results['history_simple'].history['val_loss'], label='Simple - Validation', linewidth=2)
    ax1.plot(results['history_complex'].history['loss'], label='Complex - Training', linewidth=2)
    ax1.plot(results['history_complex'].history['val_loss'], label='Complex - Validation', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Training and validation accuracy
    ax2.plot(results['history_simple'].history['accuracy'], label='Simple - Training', linewidth=2)
    ax2.plot(results['history_simple'].history['val_accuracy'], label='Simple - Validation', linewidth=2)
    ax2.plot(results['history_complex'].history['accuracy'], label='Complex - Training', linewidth=2)
    ax2.plot(results['history_complex'].history['val_accuracy'], label='Complex - Validation', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================

def main():
    """
    Main execution pipeline for ML diagnostics.
    """
    # Generate and split regression data
    X_reg, y_reg = generate_regression_data()
    X_train_reg, X_cv_reg, X_test_reg, y_train_reg, y_cv_reg, y_test_reg = split_data(X_reg, y_reg)
    
    # Generate and split classification data
    X_cls, y_cls = generate_classification_data()
    X_train_cls, X_cv_cls, X_test_cls, y_train_cls, y_cv_cls, y_test_cls = split_data(X_cls, y_cls)
    
    # 1. Demonstrate overfitting
    demonstrate_overfitting(X_train_reg, X_test_reg, y_train_reg, y_test_reg, degree=10)
    
    # 2. Bias-variance analysis
    degrees, train_errors, cv_errors = bias_variance_analysis(
        X_train_reg, X_cv_reg, y_train_reg, y_cv_reg, max_degree=9
    )
    
    # 3. Regularization analysis
    lambdas, reg_train_errors, reg_cv_errors = regularization_analysis(
        X_train_reg, X_cv_reg, y_train_reg, y_cv_reg, degree=10
    )
    
    # 4. Neural network comparison
    nn_results = compare_neural_networks(
        X_train_cls, X_cv_cls, y_train_cls, y_cv_cls, epochs=50
    )
    
    # 5. Create visualizations (display only, no file saving)
    print(f"\n" + "=" * 70)
    print("Displaying Interactive Visualizations")
    print("=" * 70)
    print("  📊 Close each plot window to continue...")
    
    # Plot bias-variance analysis
    plot_learning_curves(degrees, train_errors, cv_errors)
    
    # Plot regularization analysis
    plot_regularization_curves(lambdas, reg_train_errors, reg_cv_errors)
    
    # Plot neural network comparison
    plot_neural_network_comparison(nn_results)
    
    print(f"\n" + "=" * 70)
    print("✓ ML Diagnostics Pipeline Complete!")
    print("=" * 70)
    
    print(f"\n" + "-" * 70)
    print("Summary of Findings:")
    print("-" * 70)
    optimal_degree = degrees[np.argmin(cv_errors)]
    optimal_lambda_idx = np.argmin(reg_cv_errors)
    optimal_lambda = lambdas[optimal_lambda_idx]
    
    print(f"1. Optimal Polynomial Degree: {optimal_degree}")
    print(f"2. Optimal Regularization λ: {optimal_lambda:.0e}")
    print(f"3. High-degree polynomial shows overfitting without regularization")
    print(f"4. Neural networks: Complex model benefits from L2 regularization")
    
    print(f"\n" + "-" * 70)
    print("Key Insights:")
    print("-" * 70)
    print("• High bias (underfitting): Both training and CV errors are high")
    print("• High variance (overfitting): Large gap between training and CV errors")
    print("• Regularization helps complex models by reducing variance")
    print("• Model selection should be based on CV error, not training error")
    print("=" * 70)

if __name__ == "__main__":
    main()