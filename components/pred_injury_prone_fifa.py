"""
Author: Ronen Huang

CSE 163

This program trains a Random Forest Classifier that predicts whether a player
is injury prone or not, plots the training of the Random Forest Classifier,
and visualizes and assesses its performance.
"""

import pandas as pd
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, precision_recall_fscore_support,
    precision_recall_curve, accuracy_score, ConfusionMatrixDisplay,
    PrecisionRecallDisplay
)
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgbm


# File path of combined FIFA data from 2015 to 2022.
DATA_FILE = 'data/players_15_to_22_data.csv'


plt.rcParams['axes.labelsize'] = 'larger'
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams['legend.fontsize'] = 'large'
plt.rcParams['legend.title_fontsize'] = 'large'
plt.rcParams['figure.titlesize'] = 'x-large'


def pred_injury_prone(players_15_to_22_data_path: str) -> None:
    """
    Trains a Random Forest Classifier to predict whether a player is
    injury prone or not.
    """
    players_15_to_22_data = pd.read_csv(players_15_to_22_data_path)

    # The labels are not injury prone and injury prone.
    label = 'injury_prone'
    label_classes = ['Not Injury Prone', 'Injury Prone']

    # Non-features are specific information (club, league, nationality)
    # about a player.
    non_features = [
        'sofifa_id', 'short_name', "overall", "potential",
        'wage_eur', "height_cm", "weight_kg", 'club_team_id',
        "club_name", "league_name", 'league_level', "club_jersey_number",
        'club_loaned_from', 'club_joined', 'nationality_id',
        "nationality_name", label
    ]
    features = players_15_to_22_data.drop(columns=non_features)
    features = pd.get_dummies(features)
    labels = players_15_to_22_data[label]

    # Split the data into 80% train and 20% test.
    train_features = features[features["year"] <= 2020]
    train_labels = labels.loc[train_features.index]
    test_features = features[features["year"] >= 2021]
    test_labels = labels.loc[test_features.index]

    train_features = train_features.reset_index(drop=True)
    train_labels = train_labels.reset_index(drop=True)
    test_features = test_features.reset_index(drop=True)
    test_labels = test_labels.reset_index(drop=True)

    test_pr_auc, test_f1, test_precision, \
        test_recall, test_accuracy, test_pred_proba =\
        train_rf(
            train_features, train_labels,
            test_features, test_labels.copy(), label_classes
        )

    lgbm_pr_auc, lgbm_f1, lgbm_precision, \
        lgbm_recall, lgbm_accuracy, lgbm_pred_proba =\
        train_lgbm(
            train_features, train_labels,
            test_features, test_labels.copy(), label_classes
        )

    lr_pr_auc, lr_f1, lr_precision, lr_recall, lr_accuracy, \
        lr_pred_proba =\
        train_logistic(
            train_features, train_labels,
            test_features, test_labels.copy(), label_classes
        )

    baseline_precision, baseline_f1, baseline_recall, \
        baseline_accuracy =\
        train_first_baseline(
            train_features, train_labels,
            test_features, test_labels.copy()
        )

    # The table of test metrics.
    test_metrics = pd.DataFrame(
        {
            "PR AUC":
                [
                    baseline_precision, lr_pr_auc, lgbm_pr_auc, test_pr_auc
                ],
            "F1": [baseline_f1, lr_f1, lgbm_f1, test_f1],
            "Precision":
                [
                    baseline_precision, lr_precision,
                    lgbm_precision, test_precision
                ],
            "Recall": [baseline_recall, lr_recall, lgbm_recall, test_recall],
            "Accuracy":
                [
                    baseline_accuracy, lr_accuracy,
                    lgbm_accuracy, test_accuracy
                ]
        }, ["Baseline", "Logistic Regression", "LightGBM", "Random Forest"]
    )
    test_metrics.to_csv("tables/test_metrics.csv")

    print("Test Metrics:")
    print(test_metrics)
    print()

    test_labels[test_labels == 0] = label_classes[0]
    test_labels[test_labels == 1] = label_classes[1]

    plot_precision_recall(
        test_labels, test_pred_proba, test_precision, test_recall,
        lgbm_pred_proba, lgbm_precision, lgbm_recall,
        lr_pred_proba, lr_precision, lr_recall
    )


def train_rf(train_features: pd.DataFrame, train_labels: pd.Series,
             test_features: pd.DataFrame, test_labels: pd.DataFrame,
             label_classes: list[str]) ->\
        tuple[float, float, float, float, float, np.ndarray]:
    """
    Trains Random Forest.
    """
    # The number of trees, maximum depth, and maximum features for the Random
    # Forest Classifier are hyperparameters that most influence performance
    rf_study = optuna.create_study(
        study_name="Random Forest Optimization", direction="maximize"
    )

    # Choose best hyperparameters for Random Forest Classifier
    # with Bayesian Optimization.
    rf_study.optimize(
        lambda trial: rf_objective(
            trial, train_features, train_labels
        ), 20
    )
    rf_best_params = rf_study.best_params
    n_estimators = rf_best_params["n_estimators"]
    max_depth = rf_best_params["max_depth"]
    max_features = rf_best_params["max_features"]
    min_samples_split = rf_best_params["min_samples_split"]
    min_samples_leaf = rf_best_params["min_samples_leaf"]

    params = {
        "n_estimators": n_estimators, "max_depth": max_depth,
        "max_features": max_features, "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "class_weight": "balanced_subsample", "n_jobs": 4, "verbose": 1
    }
    best_rf = CalibratedClassifierCV(
        RandomForestClassifier(**params), method="isotonic"
    ).fit(train_features.drop(columns="year"), train_labels)
    joblib.dump(best_rf, "best_random_forest.pkl", 3)

    # Validation average precisions from training.
    train_table = rf_study.trials_dataframe()[
        [
            "params_n_estimators", "params_max_depth",
            "params_max_features", "params_min_samples_split",
            "params_min_samples_leaf", "value"
        ]
    ]
    train_table = train_table.rename(
        columns={
                    "params_n_estimators": "Number of Trees",
                    "params_max_depth": "Maximum Depth",
                    "params_max_features": "Maximum Features",
                    "params_min_samples_split": "Minimum Split",
                    "params_min_samples_leaf": "Minimum Leaf",
                    "value": "Validation PR AUC"
                }
    )
    train_table.to_csv('tables/random_forest_train_table.csv', index=False)

    plot_train_val_accuracies_rf(train_table)

    # Estimate average precision of best Random Forest Classifier
    # on future unseen data.
    test_pred_proba = best_rf.predict_proba(
        test_features.drop(columns="year")
    )[:, 1]
    test_pr_auc = average_precision_score(test_labels, test_pred_proba)
    test_precisions, test_recalls, test_thresholds =\
        precision_recall_curve(test_labels, test_pred_proba)
    test_f1s = (2 * test_precisions * test_recalls) /\
        (test_precisions + test_recalls)
    test_f1 = np.nanmax(test_f1s)
    test_precision = test_precisions[np.nanargmax(test_f1s)]
    test_recall = test_recalls[np.nanargmax(test_f1s)]
    test_pred = (
        test_pred_proba >= test_thresholds[np.nanargmax(test_f1s)]
    ).astype(int)
    test_accuracy = accuracy_score(test_labels, test_pred)

    # The table of test predictions and test labels.
    test_labels[test_labels == 0] = label_classes[0]
    test_labels[test_labels == 1] = label_classes[1]
    test_pred = pd.Series(test_pred)
    test_pred[test_pred == 0] = label_classes[0]
    test_pred[test_pred == 1] = label_classes[1]
    test_table = pd.DataFrame(
        {'injury_prone': test_labels, 'injury_prone_prob': test_pred_proba}
    )
    test_table.to_csv('tables/random_forest_test_predictions.csv', index=False)

    # Confusion matrix.
    _, ax = plt.subplots(figsize=(20, 20))
    ConfusionMatrixDisplay.from_predictions(
        test_labels, test_pred, labels=label_classes, cmap='Blues', ax=ax
    )
    plt.title('Random Forest Confusion Matrix')
    plt.savefig(
        'plots/random_forest_confusion_matrix_plot.jpg',
        dpi=500, bbox_inches="tight"
    )

    return test_pr_auc, test_f1, test_precision, test_recall, \
        test_accuracy, test_pred_proba


def rf_objective(trial: optuna.Trial, train_features: pd.DataFrame,
                 train_labels: pd.Series) -> float:
    n_estimators = trial.suggest_int("n_estimators", 100, 500)
    max_depth = trial.suggest_int("max_depth", 20, 100)
    max_features = trial.suggest_float("max_features", 0.05, 0.20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)

    params = {
        "n_estimators": n_estimators, "max_depth": max_depth,
        "max_features": max_features, "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "class_weight": "balanced_subsample", "n_jobs": 4, "verbose": 1
    }

    scores = []
    for year in train_features["year"].unique():
        train_X = train_features[train_features["year"] != year].drop(
            columns="year"
        )
        train_y = train_labels.loc[train_X.index]
        val_X = train_features[train_features["year"] == year].drop(
            columns="year"
        )
        val_y = train_labels[val_X.index]
        rf = CalibratedClassifierCV(
            RandomForestClassifier(**params), method="isotonic", cv=4
        ).fit(train_X, train_y)
        scores.append(
            average_precision_score(
                val_y, rf.predict_proba(val_X)[:, 1]
            )
        )
        print(trial.number, scores[-1])

    return np.mean(scores)


def plot_train_val_accuracies_rf(train_table: pd.DataFrame) -> None:
    """
    Plots the scatter plot of validation accuracies with respect to the set of
    hyperparameters seen during training.

    The plot is saved under the 'plots' directory as an image
    'random_forest_pr_auc_plot.jpg'.
    """
    _, ax = plt.subplots(figsize=(20, 20))
    sns.scatterplot(
        train_table, x="Number of Trees", y="Validation PR AUC",
        hue="Maximum Features", size="Maximum Depth", ax=ax,
        palette=sns.color_palette("viridis", as_cmap=True)
    )
    ax.grid()
    plt.title('Random Forest Validation PR AUC vs Number of Trees')
    plt.savefig(
        'plots/random_forest_pr_auc_plot.jpg', dpi=500, bbox_inches="tight"
    )


def train_lgbm(train_features: pd.DataFrame, train_labels: pd.Series,
               test_features: pd.DataFrame, test_labels: pd.Series,
               label_classes: list[str]) ->\
        tuple[float, float, float, float, float, np.ndarray]:
    """
    Train LightGBM Classifier.
    """
    # The number of trees, number of leaves, and learning rate for the LightGBM
    # Classifier are hyperparameters that most influence performance
    lgbm_study = optuna.create_study(
        study_name="LGBM Optimization", direction="maximize"
    )

    # Choose best hyperparameters for LightGBM Classifier
    # with Bayesian Optimization.
    lgbm_study.optimize(
        lambda trial: lgbm_objective(
            trial, train_features, train_labels
        ), 20
    )
    lgbm_best_params = lgbm_study.best_params
    max_depth = lgbm_best_params["max_depth"]
    num_leaves = 2 ** max_depth
    learning_rate = lgbm_best_params["learning_rate"]
    n_estimators = lgbm_best_params["n_estimators"]
    num_iterations = lgbm_best_params["num_iterations"]
    scale_pos_weight = lgbm_best_params["scale_pos_weight"]
    min_child_samples = lgbm_best_params["min_child_samples"]

    params = {
        "max_depth": max_depth, "num_leaves": num_leaves,
        "learning_rate": learning_rate, "n_estimators": n_estimators,
        "num_iterations": num_iterations, "scale_pos_weight": scale_pos_weight,
        "min_child_samples": min_child_samples, "n_jobs": 4
    }
    best_lgbm = CalibratedClassifierCV(
        lgbm.LGBMClassifier(**params), method="isotonic"
    ).fit(train_features.drop(columns="year"), train_labels)
    joblib.dump(best_lgbm, "best_lightgbm.pkl", 3)

    # Validation average precisions from training.
    train_table = lgbm_study.trials_dataframe()[
        [
            "params_max_depth", "params_learning_rate",
            "params_n_estimators", "params_num_iterations",
            "params_scale_pos_weight", "params_min_child_samples", "value"
        ]
    ]
    train_table = train_table.rename(
        columns={
                    "params_max_depth": "Maximum Depth",
                    "params_learning_rate": "Learning Rate",
                    "params_n_estimators": "Number of Trees",
                    "params_num_iterations": "Number of Iterations",
                    "params_scale_pos_weight": "Positive Class Weight",
                    "params_min_child_samples": "Minimum Leaf",
                    "value": "Validation PR AUC"
                }
    )
    train_table.to_csv('tables/lightgbm_train_table.csv', index=False)

    plot_train_val_accuracies_lightgbm(train_table)

    # Estimate average precision of best LightGBM Classifier
    # on future unseen data.
    test_pred_proba = best_lgbm.predict_proba(
        test_features.drop(columns="year")
    )[:, 1]
    test_pr_auc = average_precision_score(test_labels, test_pred_proba)
    test_precisions, test_recalls, test_thresholds =\
        precision_recall_curve(test_labels, test_pred_proba)
    test_f1s = (2 * test_precisions * test_recalls) /\
        (test_precisions + test_recalls)
    test_f1 = np.nanmax(test_f1s)
    test_precision = test_precisions[np.nanargmax(test_f1s)]
    test_recall = test_recalls[np.nanargmax(test_f1s)]
    test_pred = (
        test_pred_proba >= test_thresholds[np.nanargmax(test_f1s)]
    ).astype(int)
    test_accuracy = accuracy_score(test_labels, test_pred)

    # The table of test predictions and test labels.
    test_labels[test_labels == 0] = label_classes[0]
    test_labels[test_labels == 1] = label_classes[1]
    test_pred = pd.Series(test_pred)
    test_pred[test_pred == 0] = label_classes[0]
    test_pred[test_pred == 1] = label_classes[1]
    test_table = pd.DataFrame(
        {'injury_prone': test_labels, 'injury_prone_prob': test_pred_proba}
    )
    test_table.to_csv('tables/lightgbm_test_predictions.csv', index=False)

    # Confusion matrix.
    _, ax = plt.subplots(figsize=(20, 20))
    ConfusionMatrixDisplay.from_predictions(
        test_labels, test_pred, labels=label_classes, cmap='Blues', ax=ax
    )
    plt.title('LightGBM Confusion Matrix')
    plt.savefig(
        'plots/lightgbm_confusion_matrix_plot.jpg',
        dpi=500, bbox_inches="tight"
    )

    return test_pr_auc, test_f1, test_precision, test_recall, \
        test_accuracy, test_pred_proba


def lgbm_objective(trial: optuna.Trial, train_features: pd.DataFrame,
                   train_labels: pd.Series) -> float:
    max_depth = trial.suggest_int("max_depth", 5, 15)
    learning_rate = trial.suggest_float(
        "learning_rate", 5e-2, 5e-1, log=True
    )
    n_estimators = trial.suggest_int("n_estimators", 100, 500)
    num_iterations = trial.suggest_int("num_iterations", 1000, 2000)
    scale_pos_weight = trial.suggest_int("scale_pos_weight", 10, 20)
    min_child_samples = trial.suggest_int("min_child_samples", 10, 50)

    params = {
        "max_depth": max_depth, "num_leaves": 2 ** max_depth,
        "learning_rate": learning_rate, "n_estimators": n_estimators,
        "num_iterations": num_iterations, "scale_pos_weight": scale_pos_weight,
        "min_child_samples": min_child_samples, "n_jobs": 4, "verbose": -1
    }

    scores = []
    for year in train_features["year"].unique():
        train_X = train_features[train_features["year"] != year].drop(
            columns="year"
        )
        train_y = train_labels.loc[train_X.index]
        val_X = train_features[train_features["year"] == year].drop(
            columns="year"
        )
        val_y = train_labels[val_X.index]
        lg = CalibratedClassifierCV(
            lgbm.LGBMClassifier(**params), method="isotonic", cv=4
        ).fit(train_X, train_y)
        scores.append(
            average_precision_score(
                val_y, lg.predict_proba(val_X)[:, 1]
            )
        )
        print(trial.number, scores[-1])

    return np.mean(scores)


def plot_train_val_accuracies_lightgbm(train_table: pd.DataFrame) -> None:
    """
    Plots the scatter plot of validation accuracies with respect to the set of
    hyperparameters seen during training.

    The plot is saved under the 'plots' directory as an image
    'lightgbm_pr_auc_plot.jpg'.
    """
    _, ax = plt.subplots(figsize=(20, 20))
    sns.scatterplot(
        train_table, x="Number of Trees", y="Validation PR AUC",
        hue="Learning Rate", size="Maximum Depth", ax=ax,
        palette=sns.color_palette("viridis", as_cmap=True)
    )
    ax.grid()
    plt.legend()
    plt.title('LightGBM Validation PR AUC vs Number of Trees')
    plt.savefig(
        'plots/lightgbm_pr_auc_plot.jpg', dpi=500, bbox_inches="tight"
    )


def train_logistic(train_features: pd.DataFrame, train_labels: pd.Series,
                   test_features: pd.DataFrame, test_labels: pd.Series,
                   label_classes: list[str]) ->\
        tuple[float, float, float, float, float, np.ndarray]:
    """
    Logistic Regression.
    """
    # The regularization strength and l1 ratio for the Logistic
    # Regression are hyperparameters that most influence performance
    lr_study = optuna.create_study(
        study_name="Logistic Regression Optimization", direction="maximize"
    )

    # Choose best hyperparameters for Logistic Regression
    # with Bayesian Optimization.
    lr_study.optimize(
        lambda trial: lr_objective(
            trial, train_features, train_labels
        ), 20
    )
    lr_best_params = lr_study.best_params
    C = 10 ** lr_best_params["C_inverse"]
    l1_ratio = lr_best_params["l1_ratio"]

    params = {
        "penalty": "elasticnet", "C": C,
        "class_weight": "balanced", "solver": "saga",
        "max_iter": 2000, "n_jobs": 4, "l1_ratio": l1_ratio
    }
    scale_model = StandardScaler()
    train_features = scale_model.fit_transform(
        train_features.drop(columns="year")
    )
    best_lr = LogisticRegression(**params).fit(train_features, train_labels)
    joblib.dump(best_lr, "best_logistic_regression.pkl", 3)

    # Validation average precisions from training.
    train_table = lr_study.trials_dataframe()[
        [
            "params_C_inverse", "params_l1_ratio", "value"
        ]
    ]
    train_table = train_table.rename(
        columns={
                    "params_C_inverse": "C_inverse",
                    "params_l1_ratio": "L1 Ratio",
                    "value": "Validation PR AUC"
                }
    )
    train_table.to_csv(
        'tables/logistic_regression_train_table.csv', index=False
    )

    plot_train_val_accuracies_lr(train_table)

    # Estimate average precision of best Logistic Regression
    # on future unseen data.
    test_features = scale_model.transform(test_features.drop(columns="year"))
    test_pred_proba = best_lr.predict_proba(test_features)[:, 1]
    test_pr_auc = average_precision_score(test_labels, test_pred_proba)
    test_precisions, test_recalls, test_thresholds =\
        precision_recall_curve(test_labels, test_pred_proba)
    test_f1s = (2 * test_precisions * test_recalls) /\
        (test_precisions + test_recalls)
    test_f1 = np.nanmax(test_f1s)
    test_precision = test_precisions[np.nanargmax(test_f1s)]
    test_recall = test_recalls[np.nanargmax(test_f1s)]
    test_pred = (
        test_pred_proba >= test_thresholds[np.nanargmax(test_f1s)]
    ).astype(int)
    test_accuracy = accuracy_score(test_labels, test_pred)

    # The table of test predictions and test labels.
    test_labels[test_labels == 0] = label_classes[0]
    test_labels[test_labels == 1] = label_classes[1]
    test_pred = pd.Series(test_pred)
    test_pred[test_pred == 0] = label_classes[0]
    test_pred[test_pred == 1] = label_classes[1]
    test_table = pd.DataFrame(
        {'injury_prone': test_labels, 'injury_prone_prob': test_pred_proba}
    )
    test_table.to_csv(
        'tables/logistic_regression_test_predictions.csv', index=False
    )

    # Confusion matrix.
    _, ax = plt.subplots(figsize=(20, 20))
    ConfusionMatrixDisplay.from_predictions(
        test_labels, test_pred, labels=label_classes, cmap='Blues', ax=ax
    )
    plt.title('Logistic Regression Confusion Matrix')
    plt.savefig(
        'plots/logistic_regression_confusion_matrix_plot.jpg',
        dpi=500, bbox_inches="tight"
    )

    return test_pr_auc, test_f1, test_precision, test_recall, \
        test_accuracy, test_pred_proba


def lr_objective(trial: optuna.Trial, train_features: pd.DataFrame,
                 train_labels: pd.Series) -> float:
    C = 10 ** trial.suggest_float("C_inverse", -2, 2)
    l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)

    params = {
        "penalty": "elasticnet", "C": C,
        "class_weight": "balanced", "solver": "saga",
        "max_iter": 2000, "n_jobs": 4, "l1_ratio": l1_ratio
    }

    scores = []
    for year in train_features["year"].unique():
        train_X = train_features[train_features["year"] != year].drop(
            columns="year"
        )
        train_y = train_labels.loc[train_X.index]
        val_X = train_features[train_features["year"] == year].drop(
            columns="year"
        )
        val_y = train_labels[val_X.index]
        scale_model = StandardScaler()
        train_X = scale_model.fit_transform(train_X)
        val_X = scale_model.transform(val_X)
        lr = LogisticRegression(**params).fit(train_X, train_y)
        scores.append(
            average_precision_score(
                val_y, lr.predict_proba(val_X)[:, 1]
            )
        )
        print(trial.number, scores[-1])

    return np.mean(scores)


def plot_train_val_accuracies_lr(train_table: pd.DataFrame) -> None:
    """
    Plots the scatter plot of validation accuracies with respect to the set of
    hyperparameters seen during training.

    The plot is saved under the 'plots' directory as an image
    'logistic_regression_pr_auc_plot.jpg'.
    """
    _, ax = plt.subplots(figsize=(20, 20))
    sns.scatterplot(
        train_table, x="C_inverse", y="Validation PR AUC",
        hue="L1 Ratio", ax=ax,
        palette=sns.color_palette("viridis", as_cmap=True)
    )
    ax.grid()
    plt.legend()
    plt.title('Logistic Regression Validation PR AUC vs Number of Trees')
    plt.savefig(
        'plots/logistic_regression_pr_auc_plot.jpg',
        dpi=500, bbox_inches="tight"
    )


def train_first_baseline(train_features: pd.DataFrame, train_labels: pd.Series,
                         test_features: pd.DataFrame, test_labels: pd.Series)\
        -> tuple[float, float, float, float]:
    """
    Recall at 100%.
    """
    # First baseline with 100% recall.
    baseline = DummyClassifier(strategy="constant", constant=1)
    baseline.fit(train_features, train_labels)
    baseline_pred = baseline.predict(test_features)
    baseline_precision, baseline_recall, baseline_f1, _ =\
        precision_recall_fscore_support(
            test_labels, baseline_pred, average="binary"
        )
    baseline_accuracy = accuracy_score(test_labels, baseline_pred)

    return baseline_precision, baseline_f1, baseline_recall, baseline_accuracy


def plot_precision_recall(test_labels: pd.Series, test_pred_proba: np.ndarray,
                          test_precision: float, test_recall: float,
                          lgbm_pred_proba: np.ndarray, lgbm_precision: float,
                          lgbm_recall: float, lr_pred_proba: np.ndarray,
                          lr_precision: float, lr_recall: float) -> None:
    """
    The precision recall curve is saved under the "plots" directory as an
    image "precision_recall.jpg".
    """
    _, ax = plt.subplots(figsize=(20, 20))
    PrecisionRecallDisplay.from_predictions(
        test_labels, test_pred_proba, pos_label="Injury Prone",
        name="Random Forest Classifier", ax=ax, plot_chance_level=True,
        color="blue"
    )
    PrecisionRecallDisplay.from_predictions(
        test_labels, lgbm_pred_proba, pos_label="Injury Prone",
        name="LightGBM Classifier", ax=ax, plot_chance_level=True,
        color="green"
    )
    PrecisionRecallDisplay.from_predictions(
        test_labels, lr_pred_proba, pos_label="Injury Prone",
        name="Logistic Regression", ax=ax, color="purple"
    )
    ax.scatter(test_recall, test_precision, color="blue")
    ax.scatter(lgbm_recall, lgbm_precision, color="green")
    ax.scatter(lr_recall, lr_precision, color="purple")
    ax.grid()
    plt.title("Precision Recall Curve")
    plt.savefig("plots/precision_recall.jpg", dpi=500, bbox_inches="tight")
    plt.show()
    plt.close()
