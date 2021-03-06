# -*- coding: utf-8 -*-
"""Lab3_Task2_Digits_Processing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-PPc6rUylP01atB8HhuEB1g43FjHaoZ6

### CE007 : Lab 3 - Task 2: Apply algorithm on digits dataset - LabelEncoding of features: and Train test Division 80%-20%
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

ds_digit = datasets.load_digits()
print("No. of examples and features in the dataset are:", ds_digit.data.shape)
ds = pd.DataFrame(ds_digit.data)
#ds.head()

#ds.info()
#ds.describe()

"""Printing the names of features and label types of digits"""

print("Features are as follows:\n", ds_digit.data)
print("\nLabels:\n", np.unique(ds_digit.target))

"""**Splitting the dataset into the training and testing dataset. (Roll no: 7)
  => Set Random state of model equals to your roll number**
"""

training_data, testing_data, training_target, testing_target = train_test_split(ds_digit.data, ds_digit.target, test_size = 0.20, random_state = 7)

"""Creating an instance of classifier and fitting the model."""

mnb=MultinomialNB()
mnb.fit(training_data,training_target)

"""Testing the model and getting accuracy score, confusion matrix, precision score and recall score"""

# Testing
prediction_target = mnb.predict(testing_data)

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

plt.gray() 
plt.matshow(ds_digit.images[100])