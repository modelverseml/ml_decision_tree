"""
From-Scratch Decision Tree (Classification + Regression)
--------------------------------------------------------
A binary decision tree built top-down by greedy recursive splitting. At every
node we look at every feature and every candidate threshold (midpoints between
adjacent sorted feature values) and pick the split that maximises *information
gain* — the drop in impurity from parent to weighted-children.

Impurity options (matches the README):

    classification:
        gini    : 1 - sum(p_i^2)                  -- default for classification
        entropy : -sum(p_i * log2(p_i))
        error   : 1 - max(p_i)                    -- classification error
    regression:
        mse     : variance of y in the node       -- default for regression

Stopping criteria (pre-pruning) come from the sklearn-style hyperparameters:

    max_depth, min_samples_split, min_samples_leaf

At the leaf, the prediction is the majority class (classification) or the mean
of y (regression).
"""

import numpy as np
import pandas as pd


class _Node:
    """One node in the tree — either an internal split or a leaf."""

    __slots__ = ('feature', 'threshold', 'left', 'right', 'value')

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        # `value is not None` <=> leaf node.
        self.value = value


class ManualDecisionTree:

    def __init__(
        self,
        task='classification',
        criterion=None,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
    ):

        if task not in ('classification', 'regression'):
            raise ValueError("task must be 'classification' or 'regression'")

        self.task = task
        # Sensible default impurity per task.
        self.criterion = criterion or ('gini' if task == 'classification' else 'mse')
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.root = None
        self.feature_names = None

    def _impurity(self, y):
        """Impurity of a node, using the configured criterion."""

        if len(y) == 0:
            return 0.0

        if self.task == 'regression':
            # MSE inside the node == variance of y.
            return float(((y - y.mean()) ** 2).mean())

        # Classification: empirical class probabilities.
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)

        if self.criterion == 'gini':
            return 1.0 - float((p ** 2).sum())
        if self.criterion == 'entropy':
            # Add eps so log2(0) doesn't blow up — only p==0 entries hit this.
            return float(-(p * np.log2(p + 1e-12)).sum())
        if self.criterion == 'error':
            return 1.0 - float(p.max())

        raise ValueError(f"unknown criterion: {self.criterion}")

    def _best_split(self, X, y):
        """Return the (feature, threshold) pair with the highest information gain."""

        n_samples, n_features = X.shape
        parent_impurity = self._impurity(y)
        best = None  # (gain, feature, threshold, left_mask)

        for feature in range(n_features):
            values = np.unique(X[:, feature])
            if len(values) < 2:
                continue
            # Candidate thresholds: midpoints between adjacent sorted values.
            thresholds = (values[:-1] + values[1:]) / 2.0

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                n_left = left_mask.sum()
                n_right = n_samples - n_left

                # Respect min_samples_leaf on both sides.
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                weighted_child = (
                    n_left * self._impurity(y[left_mask])
                    + n_right * self._impurity(y[~left_mask])
                ) / n_samples
                gain = parent_impurity - weighted_child

                if best is None or gain > best[0]:
                    best = (gain, feature, threshold, left_mask)

        return best

    def _leaf_value(self, y):
        """Prediction stored at a leaf node."""

        if self.task == 'regression':
            return float(y.mean())
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]

    def _build(self, X, y, depth):
        """Recursively grow the tree from the rows currently in this node."""

        # Pre-pruning stop conditions — return a leaf with the local prediction.
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return _Node(value=self._leaf_value(y))

        split = self._best_split(X, y)
        # No useful split found (e.g. all features constant or no gain).
        if split is None or split[0] <= 0:
            return _Node(value=self._leaf_value(y))

        _, feature, threshold, left_mask = split
        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return _Node(feature=feature, threshold=threshold, left=left, right=right)

    def fit(self, X, y):
        """Build the tree from training data."""

        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
            X = X.to_numpy()
        else:
            X = np.asarray(X)
        y = np.asarray(y)

        self.root = self._build(X, y, depth=0)
        return self

    def _predict_one(self, x):
        """Walk down the tree until we land on a leaf."""

        node = self.root
        while node.value is None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        """Predict y for new inputs X."""

        if hasattr(X, 'to_numpy'):
            X = X.to_numpy()
        else:
            X = np.asarray(X)
        return np.array([self._predict_one(x) for x in X])

    def _depth(self, node):
        if node is None or node.value is not None:
            return 0
        return 1 + max(self._depth(node.left), self._depth(node.right))

    @property
    def depth(self):
        """Maximum depth of the fitted tree."""

        return self._depth(self.root)

    def _n_leaves(self, node):
        if node is None:
            return 0
        if node.value is not None:
            return 1
        return self._n_leaves(node.left) + self._n_leaves(node.right)

    @property
    def n_leaves(self):
        """Number of leaf nodes in the fitted tree."""

        return self._n_leaves(self.root)
