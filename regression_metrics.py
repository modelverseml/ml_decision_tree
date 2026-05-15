"""
Regression Metrics & Residual Diagnostics
-----------------------------------------
Reports the standard regression metrics (MAE, MSE, RMSE, R², Adjusted R²)
and produces residual plots used to check the linear-regression assumptions
discussed in the README:

    - Residuals vs fitted   → linearity + homoscedasticity
    - Residual histogram    → normality of errors

Adjusted R² requires the number of predictors (`n_features`); pass it in
to enable that metric.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class RegressionMetrics:

    def __init__(self, actual, predict, n_features=None):

        # Convert to numpy to avoid pandas index-alignment surprises when subtracting.
        self.actual = np.asarray(actual)
        self.predict = np.asarray(predict)
        self.n_features = n_features

    def get_metrics(self):
        """Print MAE, MSE, RMSE, R², and Adjusted R² if n_features is known."""

        mae = mean_absolute_error(self.actual, self.predict)
        mse = mean_squared_error(self.actual, self.predict)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.actual, self.predict)

        print(f"MAE  : {mae:.4f}")
        print(f"MSE  : {mse:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"R²   : {r2:.4f}")

        if self.n_features is not None:
            # Adjusted R² penalises adding features that do not improve fit.
            n = len(self.actual)
            p = self.n_features
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            print(f"Adj R² : {adj_r2:.4f}")

    def plot_residuals(self):
        """Residuals-vs-fitted and residual-distribution diagnostics."""

        residuals = self.actual - self.predict

        plt.scatter(self.predict, residuals, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Fitted")
        plt.show()

        sns.histplot(residuals, kde=True)
        plt.title("Residuals Distribution")
        plt.xlabel("Residual")
        plt.show()
