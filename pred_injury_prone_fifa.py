"""
****, ****, Ronen H

CSE 163

This program trains a Random Forest Classifier that predicts whether a player is injury prone or not,
plots the training of the Random Forest Classifier, and visualizes and assesses its performance.
"""

import pandas as pd
import optuna
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from tqdm import trange
from sklearn.tree import plot_tree
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_recall_fscore_support,\
    accuracy_score, ConfusionMatrixDisplay, PrecisionRecallDisplay
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import ComplementNB
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

# File path of combined FIFA data from 2015 to 2022.
DATA_FILE = 'data/players_15_to_22_data.csv'


def pred_injury_prone(players_15_to_22_data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, float, float,
                                                                    pd.DataFrame, pd.DataFrame, np.ndarray, float,
                                                                    float, float, pd.DataFrame, pd.DataFrame]:
    """
    Trains a Random Forest Classifier to predict whether a player is injury prone or not.

    Returns validation average precisions table, injury prone probability, and test metrics.
    """

    # The labels are not injury prone and injury prone.
    label = 'injury_prone'
    label_classes = ['Not Injury Prone', 'Injury Prone']

    # Non-features are specific information (club, league, nationality) about a player.
    non_features = ['sofifa_id', 'short_name', 'value_eur', 'club_team_id', "club_name", "league_name", 'league_level',
                    "club_jersey_number", 'club_loaned_from', 'club_joined', 'nationality_id', "nationality_name",
                    label]
    features = players_15_to_22_data.drop(columns=non_features)
    features = pd.get_dummies(features)
    labels = players_15_to_22_data[label]

    # The number of trees, maximum depth, and maximum features for the Random Forest Classifier are hyperparameters
    # that most influence performance
    rf_study = optuna.create_study(study_name="Random Forest Optimization", direction="maximize")

    # Split the data into 80% train and 20% test.
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2,
                                                                                stratify=labels)

    # Choose best hyperparameters for Random Forest Classifier with Bayesian Optimization.
    rf_study.optimize(lambda trial: rf_objective(trial, train_features, train_labels), 100, n_jobs=6)
    rf_best_params = rf_study.best_params
    best_rf = RandomForestClassifier(rf_best_params["n_estimators"], max_depth=rf_best_params["max_depth"],
                                     max_features=rf_best_params["max_features"], n_jobs=4, verbose=1)
    best_rf.fit(train_features, train_labels)
    joblib.dump(best_rf, "best_random_forest.pkl", 3)
    for i in trange(1, 25):
        _, ax = plt.subplots(figsize=(30, 30))
        plot_tree(best_rf.estimators_[i - 1], max_depth=3, feature_names=best_rf.feature_names_in_,
                  class_names=label_classes, filled=True, ax=ax)
        plt.savefig("plots/random_forest_tree_" + str(i) + ".jpg", dpi=500, bbox_inches="tight")
        plt.close()

    # Validation average precisions from training.
    train_table = rf_study.trials_dataframe()[["params_n_estimators", "params_max_depth", "params_max_features",
                                               "value"]]
    train_table = train_table.rename(columns={"params_n_estimators": "Number of Trees",
                                              "params_max_depth": "Maximum Depth",
                                              "params_max_features": "Maximum Features",
                                              "value": "Validation PR AUC"})
    train_table.to_csv('tables/random_forest_train_table.csv', index=False)

    # Estimate average precision of best Random Forest Classifier on future unseen data.
    test_pred_proba = best_rf.predict_proba(test_features)[:, 1]
    test_pr_auc = average_precision_score(test_labels, test_pred_proba)
    test_precisions, test_recalls, test_thresholds = precision_recall_curve(test_labels, test_pred_proba)
    test_f1s = (2 * test_precisions * test_recalls) / (test_precisions + test_recalls)
    test_f1 = np.nanmax(test_f1s)
    test_precision = test_precisions[np.nanargmax(test_f1s)]
    test_recall = test_recalls[np.nanargmax(test_f1s)]
    test_pred = (test_pred_proba >= test_thresholds[np.nanargmax(test_f1s)]).astype(int)
    test_accuracy = accuracy_score(test_labels, test_pred)

    # First baseline with 100% recall.
    baseline = DummyClassifier(strategy="constant", constant=1)
    baseline.fit(train_features, train_labels)
    baseline_pred = baseline.predict(test_features)
    baseline_precision, baseline_recall, baseline_f1, _ = precision_recall_fscore_support(test_labels, baseline_pred,
                                                                                          average="binary")
    baseline_accuracy = accuracy_score(test_labels, baseline_pred)

    # Second baseline with Naive Bayes.
    nb = ComplementNB(norm=True)
    nb.fit(train_features, train_labels)
    nb_pred_proba = nb.predict_proba(test_features)[:, 1]
    nb_pr_auc = average_precision_score(test_labels, nb_pred_proba)
    nb_precisions, nb_recalls, nb_thresholds = precision_recall_curve(test_labels, nb_pred_proba)
    nb_f1s = (2 * nb_precisions * nb_recalls) / (nb_precisions + nb_recalls)
    nb_f1 = np.nanmax(nb_f1s)
    nb_precision = nb_precisions[np.nanargmax(nb_f1s)]
    nb_recall = nb_recalls[np.nanargmax(nb_f1s)]
    nb_pred = (nb_pred_proba >= nb_thresholds[np.nanargmax(nb_f1s)]).astype(int)
    nb_accuracy = accuracy_score(test_labels, nb_pred)

    # The table of test metrics.
    test_metrics = pd.DataFrame({"PR AUC": [baseline_precision, nb_pr_auc, test_pr_auc],
                                 "F1": [baseline_f1, nb_f1, test_f1],
                                 "Precision": [baseline_precision, nb_precision, test_precision],
                                 "Recall": [baseline_recall, nb_recall, test_recall],
                                 "Accuracy": [baseline_accuracy, nb_accuracy, test_accuracy]},
                                ["Baseline", "Naive Bayes", "Random Forest"])
    test_metrics.to_csv("tables/test_metrics.csv")

    # The table of test predictions and test labels.
    test_labels = test_labels.reset_index(drop=True)
    test_labels[test_labels == 0] = label_classes[0]
    test_labels[test_labels == 1] = label_classes[1]
    test_pred = pd.Series(test_pred)
    test_pred[test_pred == 0] = label_classes[0]
    test_pred[test_pred == 1] = label_classes[1]
    test_table = pd.DataFrame({'Test Predictions': test_pred, 'Test Labels': test_labels})
    test_table.to_csv('tables/random_forest_test_predictions.csv', index=False)

    # The baseline table.
    baseline_pred = pd.Series(baseline_pred)
    baseline_pred[baseline_pred == 0] = label_classes[0]
    baseline_pred[baseline_pred == 1] = label_classes[1]
    baseline_table = pd.DataFrame({'Test Predictions': baseline_pred, 'Test Labels': test_labels})
    baseline_table.to_csv('tables/baseline_test_predictions.csv', index=False)

    # The Naive Bayes table.
    nb_pred = pd.Series(nb_pred)
    nb_pred[nb_pred == 0] = label_classes[0]
    nb_pred[nb_pred == 1] = label_classes[1]
    nb_table = pd.DataFrame({'Test Predictions': nb_pred, 'Test Labels': test_labels})
    nb_table.to_csv('tables/naive_bayes_test_predictions.csv', index=False)

    return train_table, test_pred_proba, test_precision, test_recall, test_table, baseline_table, nb_pred_proba,\
        nb_precision, nb_recall, nb_accuracy, nb_table, test_metrics


def rf_objective(trial: optuna.Trial, train_features: pd.DataFrame, train_labels: pd.Series) -> float:
    n_estimators = trial.suggest_int("n_estimators", 1, 1000, log=True)
    max_depth = trial.suggest_int("max_depth", 1, 1000, log=True)
    max_features = trial.suggest_float("max_features", 0.01, 0.10, log=True)

    rf = GridSearchCV(RandomForestClassifier(verbose=1), {"n_estimators": [n_estimators], "max_depth": [max_depth],
                                                          "max_features": [max_features]}, scoring="average_precision",
                      refit=False, cv=4, verbose=3)
    rf.fit(train_features, train_labels)

    return rf.best_score_


def plot_train_val_accuracies(train_table: pd.DataFrame) -> None:
    """
    Plots the scatter plot of validation accuracies with respect to the set of hyperparameters seen during training.

    The plot is saved under the 'plots' directory as an image 'random_forest_pr_auc_plot.jpg'.
    """
    _, ax = plt.subplots(figsize=(20, 20))
    sns.scatterplot(train_table, x="Number of Trees", y="Validation PR AUC", hue="Maximum Features",
                    size="Maximum Depth", ax=ax)
    ax.grid()
    plt.legend()
    plt.title('Random Forest Validation PR AUC vs Number of Trees')
    plt.savefig('plots/random_forest_pr_auc_plot.jpg', dpi=500, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_confusion_matrix(test_table: pd.DataFrame, test_pred_proba: np.ndarray, test_precision: float,
                          test_recall: float, baseline_table: pd.DataFrame, nb_table: pd.DataFrame,
                          nb_pred_proba: np.ndarray, nb_precision: float, nb_recall: float) -> None:
    """
    Plots the confusion matrix from the test predictions of the best Random Forest Classifier. Plots the precision
    recall curve from the test probabilities of the best Random Forest Classifier.

    The confusion matrix is saved under the 'plots' directory as an image 'random_forest_confusion_matrix_plot.jpg'.

    The precision recall curve is saved under the "plots" directory as an image "precision_recall.jpg".
    """

    test_pred = test_table['Test Predictions']
    test_labels = test_table['Test Labels']

    label_classes = ['Injury Prone', 'Not Injury Prone']

    _, ax = plt.subplots(figsize=(20, 20))
    ConfusionMatrixDisplay.from_predictions(test_labels, test_pred, labels=label_classes, cmap='Blues', ax=ax)
    plt.title('Random Forest Confusion Matrix')
    plt.savefig('plots/random_forest_confusion_matrix_plot.jpg', dpi=500, bbox_inches="tight")
    plt.show()
    plt.close()

    nb_pred = nb_table["Test Predictions"]

    _, ax = plt.subplots(figsize=(20, 20))
    ConfusionMatrixDisplay.from_predictions(test_labels, nb_pred, labels=label_classes, cmap='Blues', ax=ax)
    plt.title('Naive Bayes Confusion Matrix')
    plt.savefig('plots/naive_bayes_confusion_matrix_plot.jpg', dpi=500, bbox_inches="tight")
    plt.show()
    plt.close()

    baseline_pred = baseline_table["Test Predictions"]

    _, ax = plt.subplots(figsize=(20, 20))
    ConfusionMatrixDisplay.from_predictions(test_labels, baseline_pred, labels=label_classes, cmap='Blues', ax=ax)
    plt.title('Baseline Confusion Matrix')
    plt.savefig('plots/baseline_confusion_matrix_plot.jpg', dpi=500, bbox_inches="tight")
    plt.close()

    _, ax = plt.subplots(figsize=(20, 20))
    PrecisionRecallDisplay.from_predictions(test_labels, test_pred_proba, pos_label="Injury Prone",
                                            name="Random Forest Classifier", ax=ax, plot_chance_level=True,
                                            color="blue")
    PrecisionRecallDisplay.from_predictions(test_labels, nb_pred_proba, pos_label="Injury Prone",
                                            name="Naive Bayes Classifier", ax=ax, color="purple")
    ax.scatter(test_recall, test_precision, color="blue")
    ax.scatter(nb_recall, nb_precision, color="purple")
    ax.grid()
    plt.title("Precision Recall Curve")
    plt.savefig("plots/precision_recall.jpg", dpi=500, bbox_inches="tight")
    plt.show()
    plt.close()


def main():
    players_15_to_22_data = pd.read_csv(DATA_FILE)

    train_table, test_pred_proba, test_precision, test_recall, test_table, baseline_table, nb_pred_proba, nb_precision,\
        nb_recall, nb_accuracy, nb_table, test_metrics = pred_injury_prone(players_15_to_22_data)

    print('Train Table:')
    print(train_table)
    print()
    print("Test Metrics:")
    print(test_metrics)
    print()

    plot_train_val_accuracies(train_table)

    plot_confusion_matrix(test_table, test_pred_proba, test_precision, test_recall, baseline_table,
                          nb_table, nb_pred_proba, nb_precision, nb_recall)


if __name__ == '__main__':
    main()
