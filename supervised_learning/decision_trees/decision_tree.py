"""
Decision Tree Implementation from Scratch
==========================================

A complete implementation of a Decision Tree classifier using the Information Gain
approach with entropy as the splitting criterion.

Mathematical Foundation:
------------------------
1. ENTROPY:
   The entropy H(p₁) measures the impurity/uncertainty in a node:
   
   $H(p_1) = -p_1 \log_2(p_1) - (1 - p_1) \log_2(1 - p_1)$
   
   Where p₁ is the fraction of positive examples (class 1) in the node.
   
   Properties:
   - H(0) = H(1) = 0 (pure node, no uncertainty)
   - H(0.5) = 1 (maximum uncertainty, equal split)

2. INFORMATION GAIN:
   Measures the reduction in entropy after splitting on a feature:
   
   $\text{IG} = H(p_1^{\text{node}}) - \left(w^{\text{left}} H(p_1^{\text{left}}) + w^{\text{right}} H(p_1^{\text{right}})\right)$
   
   Where:
   - H(p₁ⁿᵒᵈᵉ) = entropy of current node
   - wˡᵉᶠᵗ = proportion of examples going to left branch
   - wʳⁱᵍʰᵗ = proportion of examples going to right branch
   - H(p₁ˡᵉᶠᵗ), H(p₁ʳⁱᵍʰᵗ) = entropy of left and right branches

3. DECISION TREE ALGORITHM:
   - Start at the root with all training examples
   - Calculate information gain for all features
   - Split on the feature with highest information gain
   - Recursively build left subtree (feature = 1) and right subtree (feature = 0)
   - Stop when: max depth reached, or information gain = 0, or node is pure

Author: ML Algorithms from Scratch Project
Date: January 2026
"""

import numpy as np

# =============================================================================
# SECTION 1: ENTROPY CALCULATION
# =============================================================================

def compute_entropy(y):
    """
    Computes the entropy H(p₁) for a given set of labels.
    
    The entropy formula is:
        H(p₁) = -p₁ * log₂(p₁) - (1 - p₁) * log₂(1 - p₁)
    
    Where p₁ is the fraction of examples with label 1.
    
    Parameters
    ----------
    y : array-like
        Array of binary labels (0 or 1) for the examples at this node.
        
    Returns
    -------
    entropy : float
        The entropy value. Returns 0 for empty arrays or pure nodes.
        
    Examples
    --------
    >>> compute_entropy(np.array([1, 1, 1, 1]))  # Pure node (all 1s)
    0.0
    >>> compute_entropy(np.array([0, 0, 0, 0]))  # Pure node (all 0s)
    0.0
    >>> compute_entropy(np.array([1, 1, 0, 0]))  # Maximum entropy
    1.0
    
    Mathematical Insight:
    --------------------
    - Entropy is maximized (H = 1) when p₁ = 0.5 (equal class distribution)
    - Entropy is minimized (H = 0) when p₁ = 0 or p₁ = 1 (pure node)
    - This follows from the shape of the entropy function, which is concave
    """
    # Handle edge case: empty node
    if len(y) == 0:
        return 0.0
    
    # Calculate p₁ (fraction of positive examples)
    p1 = np.mean(y)
    
    # Handle edge cases: pure nodes (p₁ = 0 or p₁ = 1)
    # log₂(0) is undefined, but lim(p→0) p*log₂(p) = 0
    if p1 == 0 or p1 == 1:
        return 0.0
    
    # Calculate entropy using the formula:
    # H(p₁) = -p₁ * log₂(p₁) - (1 - p₁) * log₂(1 - p₁)
    entropy = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)
    
    return entropy


# =============================================================================
# SECTION 2: DATASET SPLITTING
# =============================================================================

def split_dataset(X, node_indices, feature):
    """
    Splits the dataset at the given node based on a specified feature.
    
    In a binary decision tree with binary features:
    - Left branch: examples where feature value = 1
    - Right branch: examples where feature value = 0
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The data matrix containing all examples.
    node_indices : list or array
        List of indices of examples currently at this node.
    feature : int
        Index of the feature to split on.
        
    Returns
    -------
    left_indices : list
        Indices of examples going to the left branch (feature = 1).
    right_indices : list
        Indices of examples going to the right branch (feature = 0).
        
    Examples
    --------
    >>> X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    >>> split_dataset(X, [0, 1, 2, 3], feature=0)
    ([0, 2], [1, 3])  # Indices where X[:, 0] == 1 and X[:, 0] == 0
    
    Note
    ----
    This implementation assumes binary features (0 or 1).
    For continuous features, a threshold-based split would be needed.
    """
    left_indices = []
    right_indices = []
    
    # Iterate through all examples at this node
    for idx in node_indices:
        if X[idx, feature] == 1:
            # Feature value is 1 → goes to left branch
            left_indices.append(idx)
        else:
            # Feature value is 0 → goes to right branch
            right_indices.append(idx)
    
    return left_indices, right_indices


# =============================================================================
# SECTION 3: INFORMATION GAIN CALCULATION
# =============================================================================

def compute_information_gain(X, y, node_indices, feature):
    """
    Computes the information gain from splitting on a given feature.
    
    Information Gain measures the reduction in entropy achieved by splitting
    the node on a particular feature:
    
        IG = H(p₁ⁿᵒᵈᵉ) - [wˡᵉᶠᵗ · H(p₁ˡᵉᶠᵗ) + wʳⁱᵍʰᵗ · H(p₁ʳⁱᵍʰᵗ)]
    
    Where:
    - wˡᵉᶠᵗ = |left| / |node| (fraction of examples going left)
    - wʳⁱᵍʰᵗ = |right| / |node| (fraction of examples going right)
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The data matrix.
    y : ndarray of shape (n_samples,)
        The target labels (binary: 0 or 1).
    node_indices : list or array
        Indices of examples at the current node.
    feature : int
        Index of the feature to evaluate for splitting.
        
    Returns
    -------
    information_gain : float
        The information gain from splitting on this feature.
        Higher values indicate better splits.
        
    Examples
    --------
    >>> X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    >>> y = np.array([1, 1, 0, 0])
    >>> compute_information_gain(X, y, [0, 1, 2, 3], feature=0)
    1.0  # Perfect split: feature 0 perfectly separates classes
    
    Mathematical Insight:
    --------------------
    - IG ≥ 0 always (splitting cannot increase entropy)
    - IG = 0 when the split doesn't separate classes at all
    - IG = H(node) when the split perfectly separates classes
    """
    # Split the dataset on the given feature
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    
    # Get labels for current node
    y_node = y[node_indices]
    
    # Get labels for left and right branches
    y_left = y[left_indices]
    y_right = y[right_indices]
    
    # Calculate the total number of examples at this node
    n_node = len(node_indices)
    n_left = len(left_indices)
    n_right = len(right_indices)
    
    # Handle edge case: if split results in empty branch
    if n_left == 0 or n_right == 0:
        return 0.0
    
    # Calculate weights (proportions)
    w_left = n_left / n_node
    w_right = n_right / n_node
    
    # Calculate entropy of current node
    H_node = compute_entropy(y_node)
    
    # Calculate entropy of left and right branches
    H_left = compute_entropy(y_left)
    H_right = compute_entropy(y_right)
    
    # Calculate Information Gain
    # IG = H(node) - weighted average of children's entropy
    information_gain = H_node - (w_left * H_left + w_right * H_right)
    
    return information_gain


# =============================================================================
# SECTION 4: BEST SPLIT SELECTION
# =============================================================================

def get_best_split(X, y, node_indices):
    """
    Finds the best feature to split on by maximizing information gain.
    
    Iterates through all features and computes the information gain for each,
    returning the feature index that provides the maximum information gain.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The data matrix.
    y : ndarray of shape (n_samples,)
        The target labels.
    node_indices : list or array
        Indices of examples at the current node.
        
    Returns
    -------
    best_feature : int
        Index of the feature that provides maximum information gain.
        Returns -1 if no split improves the node (all IG = 0).
        
    Examples
    --------
    >>> X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    >>> y = np.array([1, 1, 0, 0])
    >>> get_best_split(X, y, [0, 1, 2, 3])
    0  # Feature 0 has perfect information gain
    
    Algorithm
    ---------
    1. For each feature f in {0, 1, ..., n_features-1}:
       - Compute IG(f) = information gain from splitting on f
    2. Return argmax(IG)
    """
    num_features = X.shape[1]
    
    best_feature = -1
    best_information_gain = 0.0
    
    # Iterate through all features
    for feature in range(num_features):
        # Calculate information gain for this feature
        info_gain = compute_information_gain(X, y, node_indices, feature)
        
        # Update best feature if this one is better
        if info_gain > best_information_gain:
            best_information_gain = info_gain
            best_feature = feature
    
    return best_feature


# =============================================================================
# SECTION 5: RECURSIVE TREE BUILDING
# =============================================================================

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth, 
                         feature_names=None, tree_structure=None):
    """
    Recursively builds a decision tree using the ID3 algorithm with Information Gain.
    
    The algorithm works as follows:
    1. Check stopping conditions (max depth or no information gain)
    2. Find the best feature to split on
    3. Split the data into left (feature=1) and right (feature=0) branches
    4. Recursively build subtrees for each branch
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The data matrix.
    y : ndarray of shape (n_samples,)
        The target labels.
    node_indices : list or array
        Indices of examples at the current node.
    branch_name : str
        Name of this branch for printing (e.g., "Root", "Left", "Right").
    max_depth : int
        Maximum depth of the tree (stopping criterion).
    current_depth : int
        Current depth in the tree (starts at 0).
    feature_names : list, optional
        Names of features for readable output.
    tree_structure : dict, optional
        Dictionary to store the tree structure.
        
    Returns
    -------
    tree_structure : dict
        A dictionary representing the tree structure with:
        - 'feature': the feature index to split on
        - 'left': left subtree (feature = 1)
        - 'right': right subtree (feature = 0)
        - 'leaf': True if this is a leaf node
        - 'prediction': predicted class for leaf nodes
        
    Stopping Conditions
    -------------------
    The recursion stops when:
    1. Maximum depth is reached (prevents overfitting)
    2. Information gain is 0 (no beneficial split exists)
    3. Node is pure (all examples have the same label)
    
    Examples
    --------
    >>> X = np.array([[1,1,1], [0,0,1], [0,1,0], [1,0,0]])
    >>> y = np.array([1, 1, 0, 0])
    >>> tree = build_tree_recursive(X, y, [0,1,2,3], "Root", 3, 0)
    """
    # Initialize tree structure if not provided
    if tree_structure is None:
        tree_structure = {}
    
    # Create indentation for tree visualization
    indent = "  " * current_depth
    
    # Get labels at current node
    y_node = y[node_indices]
    
    # Calculate current node statistics
    n_samples = len(node_indices)
    n_positive = np.sum(y_node)
    n_negative = n_samples - n_positive
    
    # Determine majority class for potential leaf prediction
    majority_class = 1 if n_positive >= n_negative else 0
    
    # Print node information
    print(f"{indent}{'─' * 2} Depth {current_depth}, {branch_name}:")
    print(f"{indent}   Samples: {n_samples} (Class 0: {n_negative}, Class 1: {n_positive})")
    
    # STOPPING CONDITION 1: Maximum depth reached
    if current_depth >= max_depth:
        print(f"{indent}   🍃 Leaf node (max depth reached) → Predict: {majority_class}")
        return {'leaf': True, 'prediction': majority_class, 'samples': n_samples}
    
    # STOPPING CONDITION 2: Pure node (all same class)
    if n_positive == 0 or n_positive == n_samples:
        prediction = 1 if n_positive == n_samples else 0
        print(f"{indent}   🍃 Leaf node (pure) → Predict: {prediction}")
        return {'leaf': True, 'prediction': prediction, 'samples': n_samples}
    
    # Find the best feature to split on
    best_feature = get_best_split(X, y, node_indices)
    
    # STOPPING CONDITION 3: No information gain from any split
    if best_feature == -1:
        print(f"{indent}   🍃 Leaf node (no info gain) → Predict: {majority_class}")
        return {'leaf': True, 'prediction': majority_class, 'samples': n_samples}
    
    # Get feature name for display
    if feature_names is not None:
        feature_display = feature_names[best_feature]
    else:
        feature_display = f"Feature {best_feature}"
    
    # Calculate and display information gain
    info_gain = compute_information_gain(X, y, node_indices, best_feature)
    print(f"{indent}   ✂️  Split on: {feature_display}")
    print(f"{indent}   📊 Information Gain: {info_gain:.4f}")
    
    # Split the dataset
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    
    print(f"{indent}   ├── Left branch ({feature_display}=1): {len(left_indices)} samples")
    print(f"{indent}   └── Right branch ({feature_display}=0): {len(right_indices)} samples")
    
    # Build tree structure
    tree_structure = {
        'leaf': False,
        'feature': best_feature,
        'feature_name': feature_display,
        'info_gain': info_gain,
        'samples': n_samples,
        'left': None,
        'right': None
    }
    
    # Recursively build left subtree (feature = 1)
    if len(left_indices) > 0:
        tree_structure['left'] = build_tree_recursive(
            X, y, left_indices, 
            f"Left ({feature_display}=1)", 
            max_depth, current_depth + 1,
            feature_names
        )
    else:
        tree_structure['left'] = {'leaf': True, 'prediction': majority_class, 'samples': 0}
    
    # Recursively build right subtree (feature = 0)
    if len(right_indices) > 0:
        tree_structure['right'] = build_tree_recursive(
            X, y, right_indices, 
            f"Right ({feature_display}=0)", 
            max_depth, current_depth + 1,
            feature_names
        )
    else:
        tree_structure['right'] = {'leaf': True, 'prediction': majority_class, 'samples': 0}
    
    return tree_structure


# =============================================================================
# SECTION 6: PREDICTION
# =============================================================================

def predict_single(x, tree):
    """
    Makes a prediction for a single example using the decision tree.
    
    Parameters
    ----------
    x : ndarray of shape (n_features,)
        A single example to classify.
    tree : dict
        The decision tree structure.
        
    Returns
    -------
    prediction : int
        The predicted class (0 or 1).
    """
    # If we've reached a leaf node, return the prediction
    if tree['leaf']:
        return tree['prediction']
    
    # Get the feature to split on
    feature = tree['feature']
    
    # Traverse left (feature=1) or right (feature=0)
    if x[feature] == 1:
        return predict_single(x, tree['left'])
    else:
        return predict_single(x, tree['right'])


def predict(X, tree):
    """
    Makes predictions for multiple examples.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The examples to classify.
    tree : dict
        The decision tree structure.
        
    Returns
    -------
    predictions : ndarray of shape (n_samples,)
        The predicted classes.
    """
    return np.array([predict_single(x, tree) for x in X])


def calculate_accuracy(y_true, y_pred):
    """
    Calculates the classification accuracy.
    
    Parameters
    ----------
    y_true : ndarray
        True labels.
    y_pred : ndarray
        Predicted labels.
        
    Returns
    -------
    accuracy : float
        Fraction of correct predictions.
    """
    return np.mean(y_true == y_pred)


# =============================================================================
# SECTION 7: DEMONSTRATION
# =============================================================================

def main():
    """
    Demonstrates the Decision Tree implementation with a sample dataset.
    
    Example Dataset: "Is this animal a cat?"
    ----------------------------------------
    Features:
    - Ear Shape: Pointy (1) / Floppy (0)
    - Face Shape: Round (1) / Not Round (0)
    - Whiskers: Present (1) / Absent (0)
    
    Target: Cat (1) / Not Cat (0)
    """
    print("=" * 70)
    print("Decision Tree from Scratch - Information Gain Approach")
    print("=" * 70)
    
    # Define the sample dataset
    # Features: [Ear Shape, Face Shape, Whiskers]
    # 1 = Pointy/Round/Present, 0 = Floppy/Not Round/Absent
    X_train = np.array([
        [1, 1, 1],  # Pointy ears, Round face, Whiskers → Cat
        [0, 0, 1],  # Floppy ears, Not round, Whiskers → Not Cat
        [0, 1, 0],  # Floppy ears, Round face, No whiskers → Not Cat
        [1, 0, 1],  # Pointy ears, Not round, Whiskers → Cat
        [1, 1, 1],  # Pointy ears, Round face, Whiskers → Cat
        [1, 1, 0],  # Pointy ears, Round face, No whiskers → Cat
        [0, 0, 0],  # Floppy ears, Not round, No whiskers → Not Cat
        [1, 1, 0],  # Pointy ears, Round face, No whiskers → Cat
        [0, 1, 1],  # Floppy ears, Round face, Whiskers → Not Cat
        [0, 1, 0],  # Floppy ears, Round face, No whiskers → Not Cat
    ])
    
    # Target: 1 = Cat, 0 = Not Cat
    y_train = np.array([1, 0, 0, 1, 1, 1, 0, 1, 0, 0])
    
    feature_names = ['Ear Shape', 'Face Shape', 'Whiskers']
    
    print("\n📋 Training Dataset:")
    print("-" * 50)
    print(f"{'Index':<6} {'Ear':<8} {'Face':<8} {'Whiskers':<10} {'Cat?'}")
    print("-" * 50)
    for i, (x, y) in enumerate(zip(X_train, y_train)):
        ear = "Pointy" if x[0] == 1 else "Floppy"
        face = "Round" if x[1] == 1 else "NotRound"
        whiskers = "Yes" if x[2] == 1 else "No"
        cat = "🐱 Yes" if y == 1 else "🐕 No"
        print(f"{i:<6} {ear:<8} {face:<8} {whiskers:<10} {cat}")
    
    print("\n" + "=" * 70)
    print("Step 1: Testing Entropy Calculation")
    print("=" * 70)
    
    # Test entropy with different distributions
    test_cases = [
        (np.array([1, 1, 1, 1]), "All 1s (pure)"),
        (np.array([0, 0, 0, 0]), "All 0s (pure)"),
        (np.array([1, 1, 0, 0]), "50-50 split"),
        (np.array([1, 1, 1, 0]), "75-25 split"),
        (y_train, "Training data"),
    ]
    
    print(f"\n{'Distribution':<20} {'Entropy H(p₁)':<15} {'p₁ value'}")
    print("-" * 50)
    for y_test, desc in test_cases:
        entropy = compute_entropy(y_test)
        p1 = np.mean(y_test)
        print(f"{desc:<20} {entropy:<15.4f} {p1:.2f}")
    
    print("\n" + "=" * 70)
    print("Step 2: Testing Dataset Splitting")
    print("=" * 70)
    
    root_indices = list(range(len(X_train)))
    print(f"\nSplitting root node (all {len(root_indices)} samples) on each feature:")
    
    for feature in range(X_train.shape[1]):
        left, right = split_dataset(X_train, root_indices, feature)
        print(f"\n  {feature_names[feature]}:")
        print(f"    Left  ({feature_names[feature]}=1): indices {left}")
        print(f"    Right ({feature_names[feature]}=0): indices {right}")
    
    print("\n" + "=" * 70)
    print("Step 3: Testing Information Gain")
    print("=" * 70)
    
    print(f"\nInformation Gain for each feature at root node:")
    print("-" * 50)
    
    for feature in range(X_train.shape[1]):
        info_gain = compute_information_gain(X_train, y_train, root_indices, feature)
        print(f"  {feature_names[feature]:<15}: IG = {info_gain:.4f}")
    
    print("\n" + "=" * 70)
    print("Step 4: Finding Best Split")
    print("=" * 70)
    
    best_feature = get_best_split(X_train, y_train, root_indices)
    print(f"\n  Best feature to split on: {feature_names[best_feature]} (index {best_feature})")
    
    print("\n" + "=" * 70)
    print("Step 5: Building the Decision Tree")
    print("=" * 70)
    print("\n🌳 Tree Structure:\n")
    
    tree = build_tree_recursive(
        X_train, y_train, 
        root_indices, 
        "Root", 
        max_depth=3, 
        current_depth=0,
        feature_names=feature_names
    )
    
    print("\n" + "=" * 70)
    print("Step 6: Making Predictions")
    print("=" * 70)
    
    # Predict on training data
    y_pred = predict(X_train, tree)
    accuracy = calculate_accuracy(y_train, y_pred)
    
    print(f"\n📊 Training Accuracy: {accuracy * 100:.1f}%")
    print(f"\n{'Index':<6} {'True':<8} {'Predicted':<10} {'Correct?'}")
    print("-" * 40)
    for i, (true, pred) in enumerate(zip(y_train, y_pred)):
        correct = "✅" if true == pred else "❌"
        print(f"{i:<6} {true:<8} {pred:<10} {correct}")
    
    # Test on new examples
    print("\n" + "=" * 70)
    print("Step 7: Predictions on New Examples")
    print("=" * 70)
    
    X_test = np.array([
        [1, 1, 1],  # Pointy, Round, Whiskers
        [0, 0, 0],  # Floppy, Not round, No whiskers
        [1, 0, 0],  # Pointy, Not round, No whiskers
    ])
    
    print("\nNew examples to classify:")
    for i, x in enumerate(X_test):
        ear = "Pointy" if x[0] == 1 else "Floppy"
        face = "Round" if x[1] == 1 else "NotRound"
        whiskers = "Yes" if x[2] == 1 else "No"
        pred = predict_single(x, tree)
        result = "🐱 Cat" if pred == 1 else "🐕 Not Cat"
        print(f"  Example {i+1}: {ear}, {face}, {whiskers} → {result}")
    
    print("\n" + "=" * 70)
    print("✅ Decision Tree Implementation Complete!")
    print("=" * 70)
    
    # Print mathematical summary
    print("\n📐 Mathematical Summary:")
    print("-" * 50)
    print("  Entropy: H(p₁) = -p₁·log₂(p₁) - (1-p₁)·log₂(1-p₁)")
    print("  Info Gain: IG = H(node) - [w_L·H(left) + w_R·H(right)]")
    print("  Split criterion: Maximize Information Gain")
    print("  Stopping: max_depth, pure node, or IG = 0")


if __name__ == "__main__":
    main()
