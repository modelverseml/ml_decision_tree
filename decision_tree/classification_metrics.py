"""
Classification Metrics & Threshold Diagnostics
----------------------------------------------
Centralises the binary-classification metrics discussed in the README:

    - Confusion matrix (with TN/FP/FN/TP)
    - Accuracy, Precision, Recall (Sensitivity), Specificity, F1
    - ROC curve + AUC
    - Precision-Recall curve
    - Cutoff sweep: accuracy / sensitivity / specificity across thresholds,
      used to pick an operating point on imbalanced data.

Usage:
    metrics = ClassificationMetrics(y_true, y_pred, y_proba)
    metrics.get_metrics()
    metrics.plot_confusion_matrix()
    metrics.plot_roc_curve()
    cutoff_df = metrics.cutoff_table()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class ClassificationMetrics:

    def __init__(self, y_true, y_pred, y_proba=None):

        # Cast to numpy to avoid pandas index-alignment surprises downstream.
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.y_proba = np.asarray(y_proba) if y_proba is not None else None

    def get_metrics(self):
        """Print the core binary-classification metrics."""

        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0

        print(f"Accuracy    : {accuracy_score(self.y_true, self.y_pred):.4f}")
        print(f"Precision   : {precision_score(self.y_true, self.y_pred, zero_division=0):.4f}")
        print(f"Recall      : {recall_score(self.y_true, self.y_pred, zero_division=0):.4f}")
        print(f"Specificity : {specificity:.4f}")
        print(f"F1 Score    : {f1_score(self.y_true, self.y_pred, zero_division=0):.4f}")
        if self.y_proba is not None:
            print(f"ROC-AUC     : {roc_auc_score(self.y_true, self.y_proba):.4f}")

    def plot_confusion_matrix(self, labels=("Negative", "Positive")):
        """Heatmap of the 2x2 confusion matrix."""

        cm = confusion_matrix(self.y_true, self.y_pred)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    def plot_roc_curve(self):
        """ROC curve with the diagonal random-guess baseline and the AUC."""

        if self.y_proba is None:
            raise ValueError("y_proba is required to plot the ROC curve.")

        fpr, tpr, _ = roc_curve(self.y_true, self.y_proba)
        auc = roc_auc_score(self.y_true, self.y_proba)

        plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random')
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Recall)")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()

    def plot_precision_recall_curve(self):
        """Precision-Recall curve — preferred over ROC for very imbalanced data."""

        if self.y_proba is None:
            raise ValueError("y_proba is required to plot the PR curve.")

        precision, recall, _ = precision_recall_curve(self.y_true, self.y_proba)
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.show()

    def cutoff_table(self, thresholds=None):
        """Accuracy / Sensitivity / Specificity across probability cutoffs.

        Use this to pick a threshold other than 0.5 — e.g. where sensitivity and
        specificity cross, which is often a better operating point on imbalanced
        data than the default 0.5.
        """

        if self.y_proba is None:
            raise ValueError("y_proba is required to compute the cutoff table.")

        if thresholds is None:
            thresholds = np.arange(0.0, 1.01, 0.1)

        rows = []
        for t in thresholds:
            preds = (self.y_proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(
                self.y_true, preds, labels=[0, 1]
            ).ravel()
            rows.append({
                'threshold': round(float(t), 2),
                'accuracy': (tp + tn) / (tp + tn + fp + fn),
                'sensitivity': tp / (tp + fn) if (tp + fn) else 0.0,
                'specificity': tn / (tn + fp) if (tn + fp) else 0.0,
            })

        df = pd.DataFrame(rows)

        plt.plot(df['threshold'], df['accuracy'], label='Accuracy')
        plt.plot(df['threshold'], df['sensitivity'], label='Sensitivity')
        plt.plot(df['threshold'], df['specificity'], label='Specificity')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Accuracy / Sensitivity / Specificity vs Threshold')
        plt.legend()
        plt.show()

        return df
