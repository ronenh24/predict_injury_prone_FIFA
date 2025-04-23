# Predict Soccer Player Injury Proneness

## Author
Ronen Huang  

## Time Frame
May 2022 to June 2022 (initial). December 2024 (improvement).  

## Data Source
The individual FIFA data from 2015 to 2022 can be accessed via [https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset).

## Pipeline
To preprocess the FIFA data for usability on machine learning tasks and predict injury proneness.
```python
from pipeline import load_clean_train

load_clean_train()
```

The combined and cleaned data is saved in the **data** directory as **[players_15_to_22_data.csv](data/players_15_to_22_data.csv)**.

The training of the Random Forest classifier with hyperparameter tuning on the number of trees, maximum depth, and maximum features is saved in the **tables** directory as **[random_forest_train_table.csv](tables/random_forest_train_table.csv)** and the scatter plot is saved in the **plots** directory as **[random_forest_pr_auc_plot.jpg](plots/random_forest_pr_auc_plot.jpg)**.
![Validation PR AUC Plot](plots/random_forest_pr_auc_plot.jpg)

The test predictions and labels of the best Random Forest classifier is saved in the **tables** directory as **[random_forest_test_predictions.csv](tables/random_forest_test_predictions.csv)** and the confusion matrix of the test performance is saved in the **plots** directory as **[random_forest_confusion_matrix_plot.jpg](plots/random_forest_confusion_matrix_plot.jpg)**.
![Confusion Matrix](plots/random_forest_confusion_matrix_plot.jpg)

The training of the LightGBM classifier with hyperparameter tuning on the learning rate, number of leaves, number of trees, and injury prone label weight is saved in the **tables** directory as **[lightgbm_train_table.csv](tables/lightgbm_train_table.csv)** and the scatter plot is saved in the **plots** directory as **[lightgbm_pr_auc_plot.jpg](plots/rlightgbm_pr_auc_plot.jpg)**.
![Validation PR AUC Plot](plots/lightgbm_pr_auc_plot.jpg)

The test predictions and labels of the best LightGBM classifier is saved in the **tables** directory as **[lightgbm_test_predictions.csv](tables/lightgbm_test_predictions.csv)** and the confusion matrix of the test performance is saved in the **plots** directory as **[lightgbm_confusion_matrix_plot.jpg](plots/lightgbm_confusion_matrix_plot.jpg)**.
![Confusion Matrix](plots/lightgbm_confusion_matrix_plot.jpg)

The test metrics of the best Random Forest classifier, best LightGBM classifier, Logistic Regression, and Baseline are saved in the **tables** directory as **[test_metrics.csv](tables/test_metrics.csv)** and the Precision Recall curve on the test data is saved in the **plots** directory as **[precision_recall.jpg](plots/precision_recall.jpg)**.
![Precision Recall Curve](plots/precision_recall.jpg)

## References
Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32. https://doi.org/10.1023/A:1010933404324

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., . . . Liu, T.-Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. In *Proceedings of the 31st International Conference on Neural Information Processing Systems* (pp. 3149–3157). Curran Associates Inc.  
