# -*- coding: utf-8 -*-
"""Lab4_Task2_Wine_Entropy.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PrwuWtEoydf77qumO6tGEZGpBV6jqq5k

### CE007 : Lab 4 - Task 2: Apply algorithm on wine dataset - LabelEncoding of features: and Train test Division 80%-20%
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

from subprocess import call

from google.colab import drive
drive.mount("/content/drive")

ds_wine = datasets.load_wine()
ds = pd.DataFrame(ds_wine.data, columns = ds_wine.feature_names)
print("No. of examples and features in the dataset are:", ds.shape)
ds.head()

#ds.info()
#ds.describe()

"""Printing the names of features and label types of digits"""

print("Features are as follows:\n", ds_wine.feature_names)
print("\nLabels:\n", np.unique(ds_wine.target_names))

"""**Splitting the dataset into the training and testing dataset. (Roll no: 7)
  => Set Random state of model equals to your roll number**
"""

training_data, testing_data, training_target, testing_target = train_test_split(ds_wine.data, ds_wine.target, test_size = 0.20, random_state = 7)

"""Creating an instance of classifier and fitting the model."""

dtc = DecisionTreeClassifier(criterion = "entropy", max_leaf_nodes = 7)
dtc.fit(training_data,training_target)

"""Testing the model and getting accuracy score, confusion matrix, precision score and recall score"""

# Testing
prediction_target = dtc.predict(testing_data)

# Getting Accuracy
accuracy = accuracy_score(testing_target, prediction_target)
print("Accuracy Score:\n", accuracy)

# Getting Confusion Matrix
cm = confusion_matrix(testing_target, prediction_target)
print("\nConfusion Matrix:\n",cm)

# Getting Precision
precision = precision_score(testing_target, prediction_target, average=None)
print("\nPrecision Score:\n", precision)

# Getting Recall
recall = recall_score(testing_target, prediction_target, average=None)
print("\nRecall Score:\n", recall)

export_graphviz(dtc, out_file='wine_tree.dot',
                feature_names=list(ds_wine.feature_names),
               class_names=list(ds_wine.target_names),
                filled=True)

# Convert to png
call(['dot', '-Tpng', 'wine_tree.dot', '-o', 'wine_tree.png', '-Gdpi=600'])

plt.figure(figsize = (15, 20))
plt.imshow(plt.imread('wine_tree.png'))
plt.axis('off')
plt.show()