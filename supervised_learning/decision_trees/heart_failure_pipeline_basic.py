"""
Heart Failure Prediction Pipeline - Basic Python Version
=======================================================

A basic implementation using only standard Python libraries.
Demonstrates the core concepts of the Trees Ensemble pipeline.

Features:
- Manual data loading and preprocessing
- Simple Decision Tree with information gain
- Hyperparameter tuning simulation
- Model evaluation

Author: ML Algorithms from Scratch Project
Date: February 2026
"""

import csv
import os
import math
import random
from collections import Counter, defaultdict

# Set random seed for reproducibility
random.seed(55)

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_heart_data(csv_path):
    """
    Load and preprocess the heart failure dataset using only standard Python.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            processed_row = {}
            for key, value in row.items():
                if key in ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'HeartDisease']:
                    processed_row[key] = int(value)
                elif key == 'Oldpeak':
                    processed_row[key] = float(value)
                else:
                    processed_row[key] = value
            data.append(processed_row)

    return data


def preprocess_data(data):
    """
    Preprocess data: one-hot encode categorical variables.
    """
    # Define categorical columns
    categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

    # Collect unique values for each categorical column
    cat_mappings = {}
    for col in categorical_cols:
        unique_vals = set(row[col] for row in data)
        cat_mappings[col] = sorted(list(unique_vals))

    # One-hot encode
    processed_data = []
    for row in data:
        new_row = {}

        # Keep numeric features
        new_row['Age'] = row['Age']
        new_row['RestingBP'] = row['RestingBP']
        new_row['Cholesterol'] = row['Cholesterol']
        new_row['FastingBS'] = row['FastingBS']
        new_row['MaxHR'] = row['MaxHR']
        new_row['Oldpeak'] = row['Oldpeak']

        # One-hot encode categorical features
        for col in categorical_cols:
            for val in cat_mappings[col]:
                new_row[f"{col}_{val}"] = 1 if row[col] == val else 0

        # Target
        new_row['HeartDisease'] = row['HeartDisease']

        processed_data.append(new_row)

    # Get feature names (excluding target)
    feature_names = [k for k in processed_data[0].keys() if k != 'HeartDisease']

    return processed_data, feature_names


def train_test_split_custom(data, test_size=0.2, random_state=55):
    """
    Custom train-test split using only standard Python.
    """
    random.seed(random_state)
    n_samples = len(data)
    indices = list(range(n_samples))
    random.shuffle(indices)

    n_test = int(n_samples * test_size)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    train_data = [data[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]

    return train_data, test_data


# =============================================================================
# DECISION TREE IMPLEMENTATION
# =============================================================================

class SimpleDecisionTree:
    """
    A simple decision tree implementation using information gain.
    """

    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, data, feature_names):
        """Fit the decision tree."""
        self.feature_names = feature_names
        self.tree = self._build_tree(data, depth=0)

    def predict(self, data):
        """Predict class labels."""
        return [self._predict_single(row) for row in data]

    def _build_tree(self, data, depth):
        """Recursively build the decision tree."""
        n_samples = len(data)

        # Extract labels
        labels = [row['HeartDisease'] for row in data]

        # Stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(set(labels)) == 1:
            return {'leaf': True, 'prediction': Counter(labels).most_common(1)[0][0]}

        # Find best split
        best_feature, best_threshold = self._find_best_split(data)

        if best_feature is None:
            return {'leaf': True, 'prediction': Counter(labels).most_common(1)[0][0]}

        # Split data
        left_data = []
        right_data = []

        for row in data:
            if row[best_feature] <= best_threshold:
                left_data.append(row)
            else:
                right_data.append(row)

        left_tree = self._build_tree(left_data, depth + 1)
        right_tree = self._build_tree(right_data, depth + 1)

        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }

    def _find_best_split(self, data):
        """Find the best feature and threshold for splitting."""
        best_gain = 0
        best_feature = None
        best_threshold = None

        labels = [row['HeartDisease'] for row in data]
        current_entropy = self._entropy(labels)
        n_samples = len(data)

        # Try each feature
        for feature in self.feature_names:
            feature_values = [row[feature] for row in data]

            # Get unique values and sort them
            unique_values = sorted(list(set(feature_values)))

            # For continuous features, try different thresholds
            if len(unique_values) > 10:
                # Use percentiles for continuous features
                sorted_values = sorted(feature_values)
                thresholds = [
                    sorted_values[int(len(sorted_values) * 0.25)],
                    sorted_values[int(len(sorted_values) * 0.5)],
                    sorted_values[int(len(sorted_values) * 0.75)]
                ]
            else:
                # Try all but last value for categorical/binary features
                thresholds = unique_values[:-1] if len(unique_values) > 1 else unique_values

            for threshold in thresholds:
                left_labels = []
                right_labels = []

                for row in data:
                    if row[feature] <= threshold:
                        left_labels.append(row['HeartDisease'])
                    else:
                        right_labels.append(row['HeartDisease'])

                if len(left_labels) == 0 or len(right_labels) == 0:
                    continue

                # Calculate information gain
                left_weight = len(left_labels) / n_samples
                right_weight = len(right_labels) / n_samples

                gain = current_entropy - (left_weight * self._entropy(left_labels) +
                                         right_weight * self._entropy(right_labels))

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _entropy(self, labels):
        """Calculate entropy of a label list."""
        if len(labels) == 0:
            return 0

        counts = Counter(labels)
        entropy = 0
        total = len(labels)

        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def _predict_single(self, row):
        """Predict a single sample."""
        node = self.tree

        while not node['leaf']:
            if row[node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']

        return node['prediction']


# =============================================================================
# RANDOM FOREST IMPLEMENTATION
# =============================================================================

class SimpleRandomForest:
    """
    A simple random forest implementation.
    """

    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, random_state=55):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []

    def fit(self, data, feature_names):
        """Fit the random forest."""
        random.seed(self.random_state)

        for i in range(self.n_estimators):
            # Bootstrap sampling
            bootstrap_data = [random.choice(data) for _ in range(len(data))]

            # Train a tree
            tree = SimpleDecisionTree(max_depth=self.max_depth,
                                    min_samples_split=self.min_samples_split)
            tree.fit(bootstrap_data, feature_names)
            self.trees.append(tree)

    def predict(self, data):
        """Predict class labels by majority voting."""
        predictions = []
        for row in data:
            votes = [tree._predict_single(row) for tree in self.trees]
            prediction = Counter(votes).most_common(1)[0][0]
            predictions.append(prediction)
        return predictions


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def accuracy_score_custom(y_true, y_pred):
    """Calculate accuracy score."""
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return correct / len(y_true)


def plot_hyperparameter_tuning(values, train_scores, val_scores, title, xlabel):
    """
    Simple text-based plotting for hyperparameter tuning.
    """
    print(f"\n{title}")
    print("-" * 60)
    print(f"{'Value':<12} {'Train Acc':<12} {'Val Acc':<12}")
    print("-" * 60)

    for val, train_acc, val_acc in zip(values, train_scores, val_scores):
        print(f"{str(val):<12} {train_acc:.4f}       {val_acc:.4f}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """
    Run the complete heart failure prediction pipeline.
    """
    print("=" * 80)
    print("HEART FAILURE PREDICTION - TREES ENSEMBLE PIPELINE (BASIC PYTHON)")
    print("=" * 80)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "heart.csv")

    # 1. Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    raw_data = load_heart_data(csv_path)
    data, feature_names = preprocess_data(raw_data)

    print(f"   Dataset size: {len(data)} samples")
    print(f"   Features: {len(feature_names)}")
    print(f"   Feature names: {feature_names[:5]}...")  # Show first 5

    # Count classes
    labels = [row['HeartDisease'] for row in data]
    class_counts = Counter(labels)
    print(f"   Classes: {sorted(class_counts.keys())}")
    print(f"   Class distribution: {dict(class_counts)}")

    # 2. Train/validation split
    print("\n2. Splitting data (80% train, 20% validation)...")
    train_data, val_data = train_test_split_custom(data, test_size=0.2, random_state=55)

    print(f"   Training set: {len(train_data)} samples")
    print(f"   Validation set: {len(val_data)} samples")

    # 3. Decision Tree hyperparameter tuning
    print("\n3. Decision Tree Hyperparameter Tuning")
    print("-" * 50)

    # Test min_samples_split
    min_samples_split_list = [2, 10, 30, 50, 100, 200, 300, 700]
    train_scores = []
    val_scores = []

    for mss in min_samples_split_list:
        tree = SimpleDecisionTree(max_depth=3, min_samples_split=mss)
        tree.fit(train_data, feature_names)

        train_pred = tree.predict(train_data)
        train_true = [row['HeartDisease'] for row in train_data]
        train_scores.append(accuracy_score_custom(train_true, train_pred))

        val_pred = tree.predict(val_data)
        val_true = [row['HeartDisease'] for row in val_data]
        val_scores.append(accuracy_score_custom(val_true, val_pred))

    plot_hyperparameter_tuning(
        min_samples_split_list,
        train_scores,
        val_scores,
        "Decision Tree: min_samples_split Tuning",
        "min_samples_split"
    )

    # Test max_depth
    max_depth_list = [1, 2, 3, 4, 8, 16, 32, 64, None]
    train_scores = []
    val_scores = []

    for depth in max_depth_list:
        tree = SimpleDecisionTree(max_depth=depth, min_samples_split=50)
        tree.fit(train_data, feature_names)

        train_pred = tree.predict(train_data)
        train_true = [row['HeartDisease'] for row in train_data]
        train_scores.append(accuracy_score_custom(train_true, train_pred))

        val_pred = tree.predict(val_data)
        val_true = [row['HeartDisease'] for row in val_data]
        val_scores.append(accuracy_score_custom(val_true, val_pred))

    plot_hyperparameter_tuning(
        [d if d is not None else "None" for d in max_depth_list],
        train_scores,
        val_scores,
        "Decision Tree: max_depth Tuning",
        "max_depth"
    )

    # Final Decision Tree
    print("\n   Final Decision Tree (min_samples_split=50, max_depth=3)")
    final_dt = SimpleDecisionTree(max_depth=3, min_samples_split=50)
    final_dt.fit(train_data, feature_names)

    train_pred = final_dt.predict(train_data)
    train_true = [row['HeartDisease'] for row in train_data]
    dt_train_acc = accuracy_score_custom(train_true, train_pred)

    val_pred = final_dt.predict(val_data)
    val_true = [row['HeartDisease'] for row in val_data]
    dt_val_acc = accuracy_score_custom(val_true, val_pred)

    print(f"   Training Accuracy:   {dt_train_acc:.4f}")
    print(f"   Validation Accuracy: {dt_val_acc:.4f}")

    # 4. Random Forest training
    print("\n4. Random Forest Training")
    print("-" * 50)

    n_estimators_list = [10, 50, 100, 500]
    rf_train_scores = []
    rf_val_scores = []

    for n_est in n_estimators_list:
        rf = SimpleRandomForest(n_estimators=n_est, max_depth=16, min_samples_split=10, random_state=55)
        rf.fit(train_data, feature_names)

        train_pred = rf.predict(train_data)
        train_true = [row['HeartDisease'] for row in train_data]
        rf_train_scores.append(accuracy_score_custom(train_true, train_pred))

        val_pred = rf.predict(val_data)
        val_true = [row['HeartDisease'] for row in val_data]
        rf_val_scores.append(accuracy_score_custom(val_true, val_pred))

    plot_hyperparameter_tuning(
        n_estimators_list,
        rf_train_scores,
        rf_val_scores,
        "Random Forest: n_estimators Tuning",
        "n_estimators"
    )

    # Final Random Forest
    print("\n   Final Random Forest (n_estimators=100, max_depth=16, min_samples_split=10)")
    final_rf = SimpleRandomForest(n_estimators=100, max_depth=16, min_samples_split=10, random_state=55)
    final_rf.fit(train_data, feature_names)

    train_pred = final_rf.predict(train_data)
    train_true = [row['HeartDisease'] for row in train_data]
    rf_train_acc = accuracy_score_custom(train_true, train_pred)

    val_pred = final_rf.predict(val_data)
    val_true = [row['HeartDisease'] for row in val_data]
    rf_val_acc = accuracy_score_custom(val_true, val_pred)

    print(f"   Training Accuracy:   {rf_train_acc:.4f}")
    print(f"   Validation Accuracy: {rf_val_acc:.4f}")

    # 5. Model Comparison
    print("\n5. Model Comparison")
    print("-" * 50)
    print(f"Decision Tree - Validation Accuracy: {dt_val_acc:.4f}")
    print(f"Random Forest  - Validation Accuracy: {rf_val_acc:.4f}")

    if rf_val_acc > dt_val_acc:
        improvement = rf_val_acc - dt_val_acc
        print(f"Random Forest performs better by {improvement:.4f} accuracy points!")
    else:
        improvement = dt_val_acc - rf_val_acc
        print(f"Decision Tree performs better by {improvement:.4f} accuracy points!")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("This demonstrates the Trees Ensemble approach using only standard Python.")
    print("=" * 80)


if __name__ == "__main__":
    main()
