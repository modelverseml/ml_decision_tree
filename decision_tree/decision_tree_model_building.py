"""
Decision Tree Model Builders
----------------------------
Thin wrappers around scikit-learn's `DecisionTreeClassifier` and
`DecisionTreeRegressor` that expose the most-tuned hyperparameters as
constructor arguments:

    - criterion           — split-quality metric (gini/entropy/log_loss for
                            classification, squared_error/friedman_mse/etc.
                            for regression). Passed through as-is; pick a
                            value appropriate to the task.
    - max_depth           — pre-pruning: hard cap on tree depth
    - min_samples_split   — minimum samples required to split an internal node
    - min_samples_leaf    — minimum samples required at a leaf node

The `parameters_dict` argument on each builder lets the notebook override the
defaults for hyperparameter tuning without rebuilding the class instance.
"""

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class DecisionTreeModels:

    def __init__(self, criterion=None, max_depth=None, min_samples_split=2, min_samples_leaf=1):

        # `criterion` defaults differ between classifier and regressor, so leave
        # it as None here and let each builder pick the right sklearn default.
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def get_classifier_model(self, X, y, parameters_dict=None):
        """Fit and return a DecisionTreeClassifier.

        Pass `parameters_dict` to override the constructor defaults (e.g. for
        grid-search results); otherwise the instance attributes are used.
        """

        if not parameters_dict:
            parameters_dict = {
                'criterion': self.criterion if self.criterion is not None else 'gini',
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
            }

        classifier_model = DecisionTreeClassifier(**parameters_dict)
        classifier_model.fit(X, y)

        return classifier_model

    def get_regressor_model(self, X, y, parameters_dict=None):
        """Fit and return a DecisionTreeRegressor.

        Pass `parameters_dict` to override the constructor defaults (e.g. for
        grid-search results); otherwise the instance attributes are used.
        """

        if not parameters_dict:
            parameters_dict = {
                'criterion': self.criterion if self.criterion is not None else 'squared_error',
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
            }

        reg_model = DecisionTreeRegressor(**parameters_dict)
        reg_model.fit(X, y)

        return reg_model
