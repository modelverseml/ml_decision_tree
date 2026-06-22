"""Decision tree toolkit.

A from-scratch binary decision tree that handles both classification and
regression, plus scikit-learn model-builder wrappers and the metric/visualisation
helpers used around them.

Public API
----------
- ManualDecisionTree   : from-scratch tree (gini / entropy / error / mse)
- DecisionTreeModels   : scikit-learn classifier/regressor builders
- TreeVisualizer       : plot a fitted tree + feature importances
- ClassificationMetrics: confusion matrix, ROC, PR, threshold sweep
- RegressionMetrics    : MAE / MSE / RMSE / R2 + residual plots

See README.md for the theory and examples/walkthrough.py for a runnable demo.
"""

from .classification_metrics import ClassificationMetrics
from .decision_tree_model_building import DecisionTreeModels
from .manual_decision_tree import ManualDecisionTree
from .regression_metrics import RegressionMetrics
from .tree_visualization import TreeVisualizer

__all__ = [
    "ManualDecisionTree",
    "DecisionTreeModels",
    "TreeVisualizer",
    "ClassificationMetrics",
    "RegressionMetrics",
]

__version__ = "1.0.0"
