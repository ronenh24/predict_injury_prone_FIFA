"""
****, ****, Ronen H

CSE 163

This programs tests the validity
of the functions for the
pred_injury_prone_fifa module.
"""

import pred_injury_prone_fifa
from unittest import TestCase
import pandas as pd

# File path of combined FIFA data from 2015 to 2022.
DATA_FILE = 'data/players_15_to_22_data.csv'


def test_compute_majority_class_inj_prone_prop(players_15_to_22_data):
    """
    Tests the correctness of the
    compute_majority_class_inj_prone_prop
    function
    """

    majority_class_prop, _ = pred_injury_prone_fifa \
        .compute_majority_class_inj_prone_prop(players_15_to_22_data)
    TestCase().assertAlmostEqual(0.9438, majority_class_prop, 4)

    return majority_class_prop


def test_pred_injury_prone(players_15_to_22_data, majority_class_prop):
    """
    Tests the validity of the
    pred_injury_prone function and therefore the
    plot_train_val_accuracies function and plot_confusion_matrix
    function that directly depends on it.
    """

    accuracies_table, best_rf, test_accuracy, test_table = \
        pred_injury_prone_fifa.pred_injury_prone(players_15_to_22_data)

    TestCase().assertLess(majority_class_prop, test_accuracy)

    TestCase().assertIsNotNone(accuracies_table)

    TestCase().assertIsNotNone(best_rf)

    TestCase().assertIsNotNone(test_table)

    best_num_trees = len(best_rf.estimators_)
    best_val_accuracy = accuracies_table['Validation Accuracy'].max()

    TestCase().assertEqual(best_val_accuracy,
                           (accuracies_table[accuracies_table[
                                'Number of Trees for Random Forest'] ==
                                best_num_trees]).iloc[0, 2])


def main():
    players_15_to_22_data = pd.read_csv(DATA_FILE)

    majority_class_prop = \
        test_compute_majority_class_inj_prone_prop(players_15_to_22_data)

    test_pred_injury_prone(players_15_to_22_data, majority_class_prop)


if __name__ == '__main__':
    main()
