# Author
Ronen H  

# Time Frame
May 2022 to June 2022 (initial). December 2024 (improvement).  

# Data Source
The individual FIFA data from 2015 to 2022 can be accessed via [https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset).

# Pipeline
To preprocess the FIFA data for usability on machine learning tasks and predict injury proneness, run the below command in powershell.
```
python pipeline.py
```

The combined and cleaned data is saved in the **data** directory as **[players_15_to_22_data.csv](data/players_15_to_22_data.csv)**.

The training of the Random Forest classifier with hyperparameter tuning on the number of trees, maximum depth, and maximum features is saved in the **tables** directory as **[random_forest_train_table.csv](tables/random_forest_train_table.csv)** and the scatter plot is saved in the **plots** directory as **[random_forest_pr_auc_plot.jpg](plots/random_forest_pr_auc_plot.jpg)**.
![Validation PR AUC Plot](plots/random_forest_pr_auc_plot.jpg)

The test predictions and labels of the best Random Forest classifier is saved in the **tables** directory as **[random_forest_test_predictions.csv](tables/random_forest_test_predictions.csv)** and the confusion matrix of the test performance is saved in the **plots** directory as **[random_forest_confusion_matrix_plot.jpg](plots/random_forest_confusion_matrix_plot.jpg)**.
![Confusion Matrix](plots/random_forest_confusion_matrix_plot.jpg)

The test metrics of the best Random Forest classifier, Naive Bayes, and Baseline are saved in the **tables** directory as **[test_metrics.csv](tables/test_metrics.csv)** and the Precision Recall curve on the test data is saved in the **plots** directory as **[precision_recall.jpg](plots/precision_recall.jpg)**.
![Precision Recall Curve](plots/precision_recall.jpg)



# References
Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32. https://doi.org/10.1023/A:1010933404324
