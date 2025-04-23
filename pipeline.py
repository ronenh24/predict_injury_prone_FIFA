"""
Author: Ronen Huang
"""

import pandas as pd
from components.load_clean_fifa_data import load_clean
from components.pred_injury_prone_fifa import pred_injury_prone, DATA_FILE


LABEL_COL = "injury_prone"
RF_PRED_PATH = "tables/random_forest_test_predictions.csv"
LGBM_PRED_PATH = "tables/lightgbm_test_predictions.csv"
LR_PRED_PATH = "tables/logistic_regression_test_predictions.csv"


def load_clean_train() -> None:
    """
    Load and Clean FIFA data. Predict Injury Proneness.
    """
    print("Load and Clean FIFA Data")
    load_clean()
    print()

    print("Predict Injury Prone")
    pred_injury_prone(DATA_FILE)

    players_15_to_22_data = pd.read_csv(DATA_FILE)
    test_data = players_15_to_22_data[
        players_15_to_22_data["year"] >= 2021
    ].drop(columns=LABEL_COL).reset_index(drop=True)

    rf_pred = pd.read_csv(RF_PRED_PATH)
    rf_pred = pd.concat([test_data, rf_pred], axis=1)
    rf_pred.to_csv(RF_PRED_PATH, index=False)

    lgbm_pred = pd.read_csv(LGBM_PRED_PATH)
    lgbm_pred = pd.concat([test_data, lgbm_pred], axis=1)
    lgbm_pred.to_csv(LGBM_PRED_PATH, index=False)

    lr_pred = pd.read_csv(LR_PRED_PATH)
    lr_pred = pd.concat([test_data, lr_pred], axis=1)
    lr_pred.to_csv(LR_PRED_PATH, index=False)
