# Author
Ronen H

# Data Source
The individual FIFA data from 2015 to 2022 can be accessed via [https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset](https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset).

# Preprocess FIFA Data
To preprocess the FIFA data for usability on machine learning tasks, on Powershell run
```
python load_clean_fifa_data.py
```
.  

The combined and cleaned data is saved in the **data** directory as **players_15_to_22_data.csv**.

# Predict Player Injury Proneness
To predict player injury proneness and to assess the performance, on Powershell run
```
python pred_injury_prone_fifa.py
```
. 

The training of the Random Forest classifier with hyperparameter tuning on the number of trees is saved in the **tables** directory as **random_forest_accuracies_table.csv** and the scatter plot is saved in the **plots** directory as **random_forest_accuracies_plot.csv**.  

The test predictions and labels of the best Random Forest classifier is saved in the **tables** directory as **random_forest_test_table.csv** and the confusion matrix of the test performance is saved in the **plots** directory as **random_forest_confusion_matrix_plot.jpg**.

# References
Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32. https://doi.org/10.1023/A:1010933404324
