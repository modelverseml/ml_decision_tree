"""
Decision Tree Visualization
---------------------------
Small helper around `sklearn.tree.plot_tree` plus a feature-importance ranking
print-out. Useful for showing the structure of a fitted classifier/regressor
in the notebook and for sanity-checking which features the tree relied on.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


class TreeVisualizer:

    def __init__(self, model, feature_names=None, class_names=None):

        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names

    def plot(self, figsize=(20, 10), max_depth=None, fontsize=10, filled=True):
        """Render the fitted tree with `sklearn.tree.plot_tree`.

        `max_depth` limits the rendered depth (the underlying tree is unchanged)
        — handy for very deep trees where the full plot becomes unreadable.
        """

        plt.figure(figsize=figsize)
        plot_tree(
            self.model,
            feature_names=self.feature_names,
            class_names=self.class_names,
            max_depth=max_depth,
            filled=filled,
            fontsize=fontsize,
        )
        plt.show()

    def feature_importance(self, top_n=None):
        """Return feature importances as a DataFrame sorted high → low.

        Pass `top_n` to keep only the top features. The DataFrame is also
        printed for quick inspection in the notebook.
        """

        importances = self.model.feature_importances_

        # Fall back to positional names if the caller didn't supply feature names.
        if self.feature_names is None:
            names = [f"feature_{i}" for i in range(len(importances))]
        else:
            names = list(self.feature_names)

        df = pd.DataFrame({
            'feature': names,
            'importance': importances,
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        if top_n is not None:
            df = df.head(top_n)

        print(df.to_string(index=False))
        return df
