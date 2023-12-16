"""
****, ****, Ronen H

CSE 163

This program trains a Random Forest Classifer that
predicts whether a player is injury prone or not,
plots the training of the Random Forest Classifier,
and visualizes and assesses its performance.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# File path of combined FIFA data from 2015 to 2022.
DATA_FILE = 'data/players_15_to_22_data.csv'


def pred_injury_prone(players_15_to_22_data):
    """
    From the given players_15_to_22_data (a pandas DataFrame)
    combined FIFA data from 2015 to 2022, trains a
    Random Forest Classifer to predict whether
    a player is injury prone or not. Returns the
    table of train accuracies and validation accuracies
    for each number of Decision Tree Classifiers
    used for the Random Forest Classifier as a
    pandas DataFrame, the best
    Random Forest Classifier based off the best
    validation accuracy, the test accuracy of the
    best Random Forest Classifier, and the table of
    test predictions from the best Random Forest
    Classifier and test labels as a pandas
    DataFrame.

    The table of train accuracies and validation
    accuracies for each number of Decision Tree
    Classifiers used for the Random Forest Classifier
    is saved under the 'tables' directory as a CSV
    'random_forest_accuracies_table.csv'.

    The table of test predictions from the
    best Random Forest Classifier and
    test labels is saved under the 'tables'
    directory as a CSV
    'random_forest_test_table.csv'.
    """

    # The label is the injury proneness of a player
    # with the two classes being not injury prone
    # and injury prone.
    label = 'injury_prone'
    label_classes = ['Not Injury Prone', 'Injury Prone']

    # Non-Features include columns with a noticeable amount of
    # missing values or 0 values.
    non_features = ['sofifa_id', 'short_name', 'value_eur',
                    'club_team_id', 'league_level', 'club_loaned_from',
                    'club_joined', 'nationality_id', label]
    features = players_15_to_22_data.drop(columns=non_features)
    features = pd.get_dummies(features)
    labels = players_15_to_22_data[label]

    # The number of Decision Tree Classifiers
    # for the Random Forest Classifier is the main
    # hyperparameter determining its performance.
    num_trees = [1, 5, 10, 15, 20, 25, 50, 75, 100]

    # Table of train accuracies for each hyperparameter setting.
    train_accuracies = []

    # Table of validation accuracies for each hyperparameter setting.
    val_accuracies = []

    # Best Random Forest Classifer based off best validation
    # accuracy.
    best_rf = None

    # The best validation accuracy.
    max_val_accuracy = -1

    # 70% of the data is used for training. 10% is used for validation.
    # 20% is used for testing.
    train_val_features, test_features, train_val_labels, test_labels = \
        train_test_split(features, labels, test_size=0.2)
    train_features, val_features, train_labels, val_labels = \
        train_test_split(train_val_features, train_val_labels, test_size=0.125)

    # Builds Random Forest Classifiers for each number of
    # Decision Tree Classifiers for the Random Forest Classifier.
    # Determines best Random Forest Classifier based
    # off best validation accuracy.
    for num_tree in num_trees:
        rf = RandomForestClassifier(num_tree,
                                    n_jobs=10,
                                    class_weight='balanced_subsample')
        rf.fit(train_features, train_labels)
        train_pred = rf.predict(train_features)
        train_accuracy = accuracy_score(train_labels, train_pred)
        train_accuracies.append(train_accuracy)
        val_pred = rf.predict(val_features)
        val_accuracy = accuracy_score(val_labels, val_pred)
        val_accuracies.append(val_accuracy)
        if val_accuracy > max_val_accuracy:
            best_rf = rf
            max_val_accuracy = val_accuracy

    # The table of train and validation accuracies
    # for each number of Decision Tree Classifiers
    # used for the Random Forest Classifier.
    accuracies_table = pd.DataFrame({
        'Number of Trees for Random Forest': num_trees,
        'Train Accuracy': train_accuracies,
        'Validation Accuracy': val_accuracies})
    accuracies_table.to_csv('tables/random_forest_accuracies_table.csv',
                            index=False)

    # Uses the test set for the best Random
    # Forest Classifier to estimate its
    # accuracy on future unseen data.
    test_pred = best_rf.predict(test_features)
    test_accuracy = accuracy_score(test_labels, test_pred)

    # The table of test predictions and test
    # labels.
    test_labels[test_labels == 0] = label_classes[0]
    test_labels[test_labels == 1] = label_classes[1]
    test_pred = pd.Series(test_pred)
    test_pred[test_pred == 0] = label_classes[0]
    test_pred[test_pred == 1] = label_classes[1]
    test_table = pd.DataFrame({
            'Test Predictions': list(test_pred),
            'Test Labels': list(test_labels)})
    test_table.to_csv('tables/random_forest_test_table.csv',
                      index=False)

    return (accuracies_table, best_rf,
            test_accuracy, test_table)


def plot_train_val_accuracies(accuracies_table):
    """
    From the given accuracies_table of train and
    validation accuracies for each number of Decision
    Tree Classifiers used for the Random Forest Classifier
    (a pandas DataFrame), plots the scatter plot with the
    accuracies as the vertical axis and
    the number of Decision Tree Classifiers
    used for the Random Forest Classifier as the
    horizontal axis.

    Essentially, the plot of the training of the
    Random Forest Classifier.

    The plot is saved under the 'plots' directory
    as an image 'random_forest_accuracies_plot.jpg'.
    """
    fig, ax = plt.subplots()
    accuracies_table.plot(x='Number of Trees for Random Forest',
                          y='Train Accuracy',
                          kind='scatter', ax=ax, color='blue')
    accuracies_table.plot(x='Number of Trees for Random Forest',
                          y='Validation Accuracy',
                          kind='scatter', ax=ax, color='orange')
    plt.legend(['Train Accuracy', 'Validation Accuracy'])
    plt.xlabel('Number of Trees for Random Forest')
    plt.ylabel('Accuracy')
    plt.title('Random Forest Accuracy vs Number of Trees for Random Forest')
    plt.savefig('plots/random_forest_accuracies_plot.jpg',
                bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(test_table):
    """
    From the given test_table (a pandas DataFrame)
    of test predictions and test labels for
    whether a player is injury prone or not,
    plots the confusion matrix.

    Can be used to assess the performance of the
    best Random Forest Classifier by
    prediction classes and label classes.

    The confusion matrix is saved under the
    'plots' directory as an image
    'random_forest_confusion_matrix_plot.jpg'.
    """

    test_pred = test_table['Test Predictions']
    test_labels = test_table['Test Labels']

    label_classes = ['Not Injury Prone', 'Injury Prone']

    ConfusionMatrixDisplay.from_predictions(test_labels, test_pred,
                                            labels=label_classes, cmap='Blues')
    plt.title('Random Forest Confusion Matrix')
    plt.savefig('plots/random_forest_confusion_matrix_plot.jpg',
                bbox_inches="tight")
    plt.show()


def compute_majority_class_inj_prone_prop(players_15_to_22_data):
    """
    From the given players_15_to_22_data (a pandas DataFrame)
    combined FIFA data from 2015 to 2022, returns the
    majority class proportion of injury prone designations
    (not injury prone and injury prone) and the majority
    class in the combined FIFA data from 2015 to 2022.

    Can be used to assess the performance of the
    best Random Forest Classifier by
    comparison to the test accuracy.
    """
    inj_prone_col = players_15_to_22_data['injury_prone']
    not_inj_prone = inj_prone_col[inj_prone_col == 0]
    inj_prone = inj_prone_col[inj_prone_col == 1]
    num_not_inj_prone = len(not_inj_prone)
    num_inj_prone = len(inj_prone)
    majority_class_prop = max(num_not_inj_prone, num_inj_prone) / \
        len(inj_prone_col)
    majority_class = "Not Injury Prone"
    if num_inj_prone > num_not_inj_prone:
        majority_class = "Injury Prone"
    return (majority_class_prop, majority_class)


def main():
    players_15_to_22_data = pd.read_csv(DATA_FILE)

    accuracies_table, best_rf, test_accuracy, test_table = \
        pred_injury_prone(players_15_to_22_data)

    print('Train Accuracy and Validation Accuracy Table:')
    print(accuracies_table)
    print()

    plot_train_val_accuracies(accuracies_table)
    print('Test Accuracy:', test_accuracy)
    print()

    print('Test Predictions and Test Labels:')
    print(test_table)
    print()

    plot_confusion_matrix(test_table)

    majority_class_prop, majority_class = \
        compute_majority_class_inj_prone_prop(players_15_to_22_data)
    print('Majority Class Proportion:', majority_class_prop)
    print('The Majority Class is', majority_class)


if __name__ == '__main__':
    main()
