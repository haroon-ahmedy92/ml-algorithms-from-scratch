"""
Binary Decision Tree - Mushroom Classification
===============================================

A complete implementation of a Binary Decision Tree for mushroom classification
using Information Gain criterion based on entropy.

Problem Statement:
------------------
Classify mushrooms as either EDIBLE (1) or POISONOUS (0) based on three features:
1. Brown Cap (1 = Brown, 0 = Not Brown)
2. Tapering Stalk (1 = Tapering, 0 = Not Tapering)
3. Solitary (1 = Solitary, 0 = Clustered)

Mathematical Foundation:
------------------------
1. ENTROPY:
   H(p₁) = -p₁ * log₂(p₁) - (1 - p₁) * log₂(1 - p₁)
   
   Where p₁ is the fraction of examples with label 1 (Edible).
   
   Special Cases:
   - If p₁ = 0 or p₁ = 1, entropy = 0 (pure node)
   - If p₁ = 0.5, entropy = 1 (maximum impurity)

2. INFORMATION GAIN:
   IG = H(p₁_node) - (w_left * H(p₁_left) + w_right * H(p₁_right))
   
   Where:
   - H(p₁_node) = entropy of the current node
   - w_left = |left| / |node| (proportion going left)
   - w_right = |right| / |node| (proportion going right)
   - H(p₁_left), H(p₁_right) = entropy of left and right branches

3. DECISION RULE:
   - For each feature, calculate the information gain from splitting on it
   - Choose the feature that maximizes information gain
   - Split data: left branch (feature = 1), right branch (feature = 0)
   - Recursively build subtrees until stopping criteria are met

Author: ML Algorithms from Scratch Project
Date: February 2026
"""

import numpy as np
from numpy import log2

# =============================================================================
# SECTION 1: ENTROPY CALCULATION
# =============================================================================

def compute_entropy(y):
    """
    Computes the entropy H(p₁) for a given set of labels.
    
    Formula: H(p₁) = -p₁ * log₂(p₁) - (1 - p₁) * log₂(1 - p₁)
    
    Parameters
    ----------
    y : array-like
        Array of binary labels (0 or 1).
        
    Returns
    -------
    entropy : float
        Entropy value in range [0, 1].
        - 0 when all examples have the same label (pure node)
        - 1 when exactly half are 0 and half are 1 (maximum impurity)
        
    Examples
    --------
    >>> compute_entropy(np.array([1, 1, 1, 1]))
    0.0  # Pure node (all positive)
    >>> compute_entropy(np.array([1, 1, 0, 0]))
    1.0  # Maximum entropy (balanced)
    """
    # Edge case: empty array
    if len(y) == 0:
        return 0.0
    
    # Calculate p1 (fraction of positive examples)
    p1 = np.mean(y)
    
    # Edge case: pure node (all 0s or all 1s)
    if p1 == 0.0 or p1 == 1.0:
        return 0.0
    
    # Calculate entropy
    entropy = -p1 * log2(p1) - (1 - p1) * log2(1 - p1)
    
    return entropy


# =============================================================================
# SECTION 2: DATASET SPLITTING
# =============================================================================

def split_dataset(X, node_indices, feature):
    """
    Splits the dataset based on a feature value.
    
    Binary split:
    - Left branch: feature value = 1
    - Right branch: feature value = 0
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data matrix where X[i, j] is the j-th feature of i-th sample.
    node_indices : list or array
        Indices of examples currently at this node.
    feature : int
        Index of the feature to split on.
        
    Returns
    -------
    left_indices : list
        Indices where X[:, feature] == 1
    right_indices : list
        Indices where X[:, feature] == 0
        
    Example
    -------
    >>> X = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 0]])
    >>> split_dataset(X, [0, 1, 2, 3], feature=0)
    ([0, 2], [1, 3])  # Samples with feature 0 = 1 and feature 0 = 0
    """
    left_indices = []
    right_indices = []
    
    for idx in node_indices:
        if X[idx, feature] == 1:
            left_indices.append(idx)
        else:
            right_indices.append(idx)
    
    return left_indices, right_indices


# =============================================================================
# SECTION 3: INFORMATION GAIN CALCULATION
# =============================================================================

def compute_information_gain(X, y, node_indices, feature):
    """
    Computes the information gain from splitting on a feature.
    
    Formula:
        IG = H(p₁_node) - (w_left * H(p₁_left) + w_right * H(p₁_right))
    
    Where:
    - H(p₁_node) = entropy at current node
    - w_left = proportion of samples going to left branch
    - w_right = proportion of samples going to right branch
    - H(p₁_left), H(p₁_right) = entropy of left and right branches
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data matrix.
    y : ndarray of shape (n_samples,)
        Binary target labels.
    node_indices : list or array
        Indices of examples at the current node.
    feature : int
        Index of the feature to evaluate.
        
    Returns
    -------
    information_gain : float
        Information gain value (≥ 0).
        - 0 means the split doesn't reduce entropy
        - Higher values indicate better splits
        
    Example
    -------
    >>> X = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
    >>> y = np.array([1, 1, 0, 0])
    >>> compute_information_gain(X, y, [0, 1, 2, 3], feature=0)
    1.0  # Perfect split: feature 0 separates all 1s from all 0s
    """
    # Split the dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    
    # Handle case where split produces empty branch
    if len(left_indices) == 0 or len(right_indices) == 0:
        return 0.0
    
    # Get labels for each branch
    y_node = y[node_indices]
    y_left = y[left_indices]
    y_right = y[right_indices]
    
    # Calculate sizes
    n_node = len(node_indices)
    n_left = len(left_indices)
    n_right = len(right_indices)
    
    # Calculate weights
    w_left = n_left / n_node
    w_right = n_right / n_node
    
    # Calculate entropy values
    H_node = compute_entropy(y_node)
    H_left = compute_entropy(y_left)
    H_right = compute_entropy(y_right)
    
    # Calculate information gain
    information_gain = H_node - (w_left * H_left + w_right * H_right)
    
    return information_gain


# =============================================================================
# SECTION 4: BEST SPLIT SELECTION
# =============================================================================

def get_best_split(X, y, node_indices):
    """
    Finds the feature that maximizes information gain.
    
    Algorithm:
    1. For each feature f in {0, 1, ..., n_features-1}:
       - Calculate IG(f) = information gain from splitting on f
    2. Return the feature with maximum IG
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data matrix.
    y : ndarray of shape (n_samples,)
        Binary target labels.
    node_indices : list or array
        Indices of examples at the current node.
        
    Returns
    -------
    best_feature : int
        Index of the feature with highest information gain.
        Returns -1 if no feature improves the split (all IG = 0).
        
    Example
    -------
    >>> X = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
    >>> y = np.array([1, 1, 0, 0])
    >>> get_best_split(X, y, [0, 1, 2, 3])
    0  # Feature 0 has the highest information gain
    """
    num_features = X.shape[1]
    
    best_feature = -1
    best_information_gain = 0.0
    
    # Test each feature
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
                        feature_names=None):
    """
    Recursively builds a decision tree using Information Gain criterion.
    
    Algorithm:
    1. Check stopping criteria:
       - If current_depth >= max_depth → Create leaf node
       - If node is pure (all same label) → Create leaf node
    2. Find best feature using get_best_split
    3. If no best feature (IG = 0) → Create leaf node
    4. Split dataset and recursively build left and right subtrees
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data matrix.
    y : ndarray of shape (n_samples,)
        Binary target labels.
    node_indices : list or array
        Indices of examples at the current node.
    branch_name : str
        Name/description of this branch (for printing).
    max_depth : int
        Maximum depth of the tree.
    current_depth : int
        Current depth (starts at 0 for root).
    feature_names : list, optional
        Names of features for readable output.
        
    Returns
    -------
    None
        The function prints the tree structure as it's being built.
        
    Stopping Criteria:
    ------------------
    1. current_depth >= max_depth (depth limit reached)
    2. Node is pure (all labels are 0 or all are 1)
    3. No feature improves the split (all IG = 0)
    
    Tree Visualization:
    -------------------
    The function prints the tree structure in a readable format showing:
    - Depth level
    - Branch name (Root, Left, Right)
    - Number of samples and class distribution
    - Feature chosen for split (if not a leaf)
    - Information gain value
    """
    # Create indentation for visualization
    indent = "  " * current_depth
    
    # Get labels at this node
    y_node = y[node_indices]
    
    # Calculate node statistics
    n_samples = len(node_indices)
    n_positive = np.sum(y_node)
    n_negative = n_samples - n_positive
    
    # Determine majority class for leaf prediction
    majority_class = 1 if n_positive >= n_negative else 0
    
    # Print node information
    print(f"{indent}Depth {current_depth}, {branch_name}:")
    print(f"{indent}  Samples: {n_samples} | Edible (1): {n_positive} | Poisonous (0): {n_negative}")
    
    # STOPPING CRITERION 1: Maximum depth reached
    if current_depth >= max_depth:
        label = "🍄 EDIBLE" if majority_class == 1 else "☠️  POISONOUS"
        print(f"{indent}  └─ LEAF (max depth reached) → Predict: {label}")
        return
    
    # STOPPING CRITERION 2: Pure node (all same label)
    if n_positive == 0 or n_positive == n_samples:
        label = "🍄 EDIBLE" if (n_positive == n_samples) else "☠️  POISONOUS"
        print(f"{indent}  └─ LEAF (pure node) → Predict: {label}")
        return
    
    # Find best feature to split on
    best_feature = get_best_split(X, y, node_indices)
    
    # STOPPING CRITERION 3: No feature improves the split
    if best_feature == -1:
        label = "🍄 EDIBLE" if majority_class == 1 else "☠️  POISONOUS"
        print(f"{indent}  └─ LEAF (no info gain) → Predict: {label}")
        return
    
    # Get feature name
    if feature_names is not None:
        feature_name = feature_names[best_feature]
    else:
        feature_name = f"Feature {best_feature}"
    
    # Calculate information gain for this split
    info_gain = compute_information_gain(X, y, node_indices, best_feature)
    
    # Print split information
    print(f"{indent}  ✂️  Split on: {feature_name}")
    print(f"{indent}  📊 Information Gain: {info_gain:.4f}")
    
    # Split the dataset
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    
    print(f"{indent}  ├─ LEFT ({feature_name}=1): {len(left_indices)} samples")
    print(f"{indent}  └─ RIGHT ({feature_name}=0): {len(right_indices)} samples")
    print()
    
    # Recursively build left subtree
    if len(left_indices) > 0:
        build_tree_recursive(
            X, y, left_indices,
            f"LEFT ({feature_name}=1)",
            max_depth, current_depth + 1,
            feature_names
        )
    
    # Recursively build right subtree
    if len(right_indices) > 0:
        build_tree_recursive(
            X, y, right_indices,
            f"RIGHT ({feature_name}=0)",
            max_depth, current_depth + 1,
            feature_names
        )


# =============================================================================
# SECTION 6: MAIN DEMONSTRATION
# =============================================================================

def main():
    """
    Demonstrates the Decision Tree implementation on mushroom classification.
    
    Dataset: 10 mushroom samples with 3 binary features
    Features:
    - Feature 0: Brown Cap (1 = Brown, 0 = Not Brown)
    - Feature 1: Tapering Stalk (1 = Tapering, 0 = Not Tapering)
    - Feature 2: Solitary (1 = Solitary, 0 = Clustered)
    
    Target:
    - 1 = Edible mushroom
    - 0 = Poisonous mushroom
    """
    
    print("=" * 80)
    print("BINARY DECISION TREE - MUSHROOM CLASSIFICATION")
    print("=" * 80)
    
    # Define the mushroom dataset
    # Features: [Brown Cap, Tapering Stalk, Solitary]
    X_train = np.array([
        [1, 1, 1],  # Brown cap, Tapering stalk, Solitary
        [1, 1, 0],  # Brown cap, Tapering stalk, Clustered
        [1, 0, 1],  # Brown cap, Not tapering, Solitary
        [0, 1, 1],  # Not brown, Tapering stalk, Solitary
        [0, 1, 0],  # Not brown, Tapering stalk, Clustered
        [1, 1, 1],  # Brown cap, Tapering stalk, Solitary
        [0, 0, 0],  # Not brown, Not tapering, Clustered
        [1, 0, 0],  # Brown cap, Not tapering, Clustered
        [0, 1, 1],  # Not brown, Tapering stalk, Solitary
        [1, 1, 1],  # Brown cap, Tapering stalk, Solitary
    ])
    
    # Target: 1 = Edible, 0 = Poisonous
    y_train = np.array([1, 1, 1, 1, 0, 1, 0, 0, 0, 1])
    
    feature_names = ['Brown Cap', 'Tapering Stalk', 'Solitary']
    
    # Display the dataset
    print("\n📊 MUSHROOM DATASET:")
    print("-" * 80)
    print(f"{'Index':<6} {'Brown Cap':<12} {'Tap. Stalk':<12} {'Solitary':<12} {'Type':<15}")
    print("-" * 80)
    
    for i, (X, y) in enumerate(zip(X_train, y_train)):
        brown = "Yes" if X[0] == 1 else "No"
        taper = "Yes" if X[1] == 1 else "No"
        solitary = "Yes" if X[2] == 1 else "No"
        mushroom_type = "🍄 Edible" if y == 1 else "☠️  Poisonous"
        print(f"{i:<6} {brown:<12} {taper:<12} {solitary:<12} {mushroom_type:<15}")
    
    print("\n" + "=" * 80)
    print("STEP 1: ENTROPY CALCULATION")
    print("=" * 80)
    
    # Calculate entropy at root
    root_entropy = compute_entropy(y_train)
    n_positive = np.sum(y_train)
    n_total = len(y_train)
    p1 = n_positive / n_total
    
    print(f"\nEntropy at Root Node:")
    print(f"  Total samples: {n_total}")
    print(f"  Edible (1): {n_positive}")
    print(f"  Poisonous (0): {n_total - n_positive}")
    print(f"  p₁ (fraction of edible): {p1:.2f}")
    print(f"  H(p₁) = {root_entropy:.4f}")
    print(f"\n  Formula: H({p1:.2f}) = -{p1:.2f}·log₂({p1:.2f}) - {1-p1:.2f}·log₂({1-p1:.2f})")
    print(f"           = {root_entropy:.4f}")
    
    print("\n" + "=" * 80)
    print("STEP 2: INFORMATION GAIN FOR EACH FEATURE")
    print("=" * 80)
    
    root_indices = list(range(len(X_train)))
    
    print(f"\nTesting each feature at the root node:")
    print("-" * 80)
    
    for feature in range(X_train.shape[1]):
        info_gain = compute_information_gain(X_train, y_train, root_indices, feature)
        print(f"  {feature_names[feature]:<15}: Information Gain = {info_gain:.4f}")
    
    print("\n" + "=" * 80)
    print("STEP 3: BUILDING THE DECISION TREE (max_depth = 2)")
    print("=" * 80)
    print("\n🌳 TREE STRUCTURE:\n")
    
    # Build the tree with max_depth = 2
    build_tree_recursive(
        X_train, y_train,
        root_indices,
        "ROOT",
        max_depth=2,
        current_depth=0,
        feature_names=feature_names
    )
    
    print("\n" + "=" * 80)
    print("✅ DECISION TREE BUILDING COMPLETE")
    print("=" * 80)
    
    print("\n📐 Mathematical Summary:")
    print("-" * 80)
    print("  Entropy: H(p₁) = -p₁·log₂(p₁) - (1-p₁)·log₂(1-p₁)")
    print("  Info Gain: IG = H(node) - [w_L·H(left) + w_R·H(right)]")
    print("  Split Criterion: Maximize Information Gain")
    print("  Stopping: max_depth reached or node is pure or IG = 0")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
