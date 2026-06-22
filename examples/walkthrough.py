"""End-to-end decision tree walkthrough.

Replaces the old notebook with a runnable script in two parts:

    Part A - Classification
      1. build an imbalanced synthetic dataset, split it
      2. fit the from-scratch tree with three impurity criteria and check the
         accuracy against scikit-learn
      3. fit the sklearn wrapper, tune it with GridSearchCV
      4. report the full classification metrics for the tuned model
      5. show the top feature importances

    Part B - Regression
      1. build a synthetic regression dataset, split it
      2. compare the from-scratch regression tree against scikit-learn
      3. tune an sklearn regressor, report regression metrics + importances

Run from the repository root:

    python examples/walkthrough.py            # print the full walkthrough
    python examples/walkthrough.py --plot     # also show the tree / metric plots
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from _helpers import add_repo_root_to_path

add_repo_root_to_path()

from decision_tree import (  # noqa: E402  (import after the sys.path tweak)
    ClassificationMetrics,
    DecisionTreeModels,
    ManualDecisionTree,
    RegressionMetrics,
    TreeVisualizer,
)

RANDOM_STATE = 42


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# --------------------------------------------------------------------------- #
# Part A: classification
# --------------------------------------------------------------------------- #
def classification_walkthrough(show_plots):
    section("PART A - CLASSIFICATION")

    # Imbalanced dataset (~30% positive) so the metrics below are meaningful.
    X, y = make_classification(
        n_samples=1500, n_features=10, n_informative=6, n_redundant=2,
        n_classes=2, weights=[0.7, 0.3], flip_y=0.03, random_state=RANDOM_STATE,
    )
    feature_names = [f"feat_{i}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name="target")
    print(f"Dataset: X={X.shape}, positive rate = {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, stratify=y, random_state=RANDOM_STATE
    )

    section("From-scratch tree vs scikit-learn (three impurity criteria)")
    rows = []
    for criterion in ["gini", "entropy", "error"]:
        manual = ManualDecisionTree(
            task="classification", criterion=criterion, max_depth=5
        ).fit(X_train, y_train)
        manual_acc = accuracy_score(y_test, manual.predict(X_test))

        if criterion in ("gini", "entropy"):
            sk = DecisionTreeClassifier(
                criterion=criterion, max_depth=5, random_state=RANDOM_STATE
            ).fit(X_train, y_train)
            sk_acc = accuracy_score(y_test, sk.predict(X_test))
        else:
            sk_acc = float("nan")  # sklearn has no 'error' criterion

        rows.append({"criterion": criterion, "manual_acc": round(manual_acc, 4),
                     "sklearn_acc": round(sk_acc, 4)})
    print(pd.DataFrame(rows).to_string(index=False))

    section("Hyperparameter tuning with GridSearchCV")
    trees = DecisionTreeModels(criterion="gini", max_depth=5)
    trees.get_classifier_model(X_train, y_train)  # baseline fit via the wrapper
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 10, 20],
        "min_samples_leaf": [1, 5, 10],
    }
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE), param_grid, cv=5, n_jobs=-1
    )
    grid.fit(X_train, y_train)
    best_clf = grid.best_estimator_
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.4f}")

    section("Classification metrics (tuned model on the test set)")
    y_pred = best_clf.predict(X_test)
    y_proba = best_clf.predict_proba(X_test)[:, 1]
    metrics = ClassificationMetrics(y_test, y_pred, y_proba)
    metrics.get_metrics()

    section("Top feature importances")
    viz = TreeVisualizer(best_clf, feature_names=feature_names, class_names=["neg", "pos"])
    viz.feature_importance(top_n=5)

    if show_plots:
        section("Classification plots")
        metrics.plot_confusion_matrix()
        metrics.plot_roc_curve()
        viz.plot(max_depth=3)


# --------------------------------------------------------------------------- #
# Part B: regression
# --------------------------------------------------------------------------- #
def regression_walkthrough(show_plots):
    section("PART B - REGRESSION")

    X, y = make_regression(
        n_samples=1500, n_features=8, n_informative=5, noise=15.0,
        random_state=RANDOM_STATE,
    )
    feature_names = [f"feat_{i}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    section("From-scratch regression tree vs scikit-learn")
    manual = ManualDecisionTree(task="regression", max_depth=5).fit(X_train, y_train)
    sk = DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE).fit(X_train, y_train)
    print(f"Manual  R2: {r2_score(y_test, manual.predict(X_test)):.4f}")
    print(f"sklearn R2: {r2_score(y_test, sk.predict(X_test)):.4f}")

    section("Hyperparameter tuning with GridSearchCV")
    param_grid = {
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 10, 20],
        "min_samples_leaf": [1, 5, 10],
    }
    grid = GridSearchCV(
        DecisionTreeRegressor(random_state=RANDOM_STATE), param_grid, cv=5,
        n_jobs=-1, scoring="r2",
    )
    grid.fit(X_train, y_train)
    best_reg = grid.best_estimator_
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV R2:  {grid.best_score_:.4f}")

    section("Regression metrics (tuned model on the test set)")
    RegressionMetrics(
        y_test, best_reg.predict(X_test), n_features=X_test.shape[1]
    ).get_metrics()

    section("Top feature importances")
    viz = TreeVisualizer(best_reg, feature_names=feature_names)
    viz.feature_importance(top_n=5)

    if show_plots:
        section("Regression tree plot")
        viz.plot(max_depth=3)


def main():
    parser = argparse.ArgumentParser(description="Run the decision tree walkthrough.")
    parser.add_argument("--plot", action="store_true", help="show the tree / metric plots")
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)
    classification_walkthrough(show_plots=args.plot)
    regression_walkthrough(show_plots=args.plot)
    print()


if __name__ == "__main__":
    main()
