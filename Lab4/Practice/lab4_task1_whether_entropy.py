# -*- coding: utf-8 -*-
"""Lab4_Task1_Whether_Entropy.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nrd08cP7UQ761PI0ud9V0AhVKqY12qIH

### CE007 : Lab 4 - Task 1: Try the algo on Dataset1 - OneHotEncoding of features: and Train test Division 70%-30%
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

from subprocess import call

from google.colab import drive
drive.mount("/content/drive")

ds1 = pd.read_csv(r"./Dataset1.csv")
print("No. of examples and features in the dataset are:", ds1.shape)
ds1.head()

ds1.info()
ds1.describe()

print("Unique Outlook values: ", ds1['Outlook'].unique())
print("Unique Temperature values: ", ds1['Temp'].unique())
print("Unique Humidity values: ", ds1['Humidity'].unique())
print("Unique Wind values: ", ds1['Wind'].unique())
print("'Play' Class Labels values: ", ds1['Play'].unique())

pie_ch = ds1['Play'].value_counts().plot.pie(autopct = "%1.1f%%", radius = 1, startangle = 0)
pie_ch.set_ylabel('')
plt.show()

dummy_ds1 = pd.get_dummies(ds1.iloc[:,:-1])

cols = ds1.columns.tolist()
cols.remove("Play")

ds1 = ds1.drop(cols, axis = 1)
ds1 = pd.concat([dummy_ds1,ds1], axis = 1)

print("\nDataset 1 after concat:\n")
ds1.head()

print("\nLabels\n")
labels = ds1['Play']
print(labels)

"""**Splitting the dataset into the training and testing dataset. (Roll no: 7)
  => Set Random state of model equals to your roll number**
"""

training_data, testing_data, training_target, testing_target = train_test_split(ds1.iloc[:,:-1], labels, test_size = 0.30, random_state = 7)

"""Creating an instance of classifier and fitting the model."""

dtc = DecisionTreeClassifier(max_leaf_nodes = 7, random_state = 7)
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

"""**(1) What will be the value of Play, if Outlook is ’Rainy’, Temperature is ’Mild’, Humidity =’Normal’, and Wind = ’False’?**

**(2) What will be the value of Play, if Outlook is ’Sunny’, Temeprature is ’Cool’, Humidity =’High’, and Wind = ’True’?**
"""

'''
Rainy = 0,1,0
Mild = 0,0,1
Normal = 0,0,1
False = 1,0
'''
case1 = [0,1,0,0,0,1,0,0,1,1,0]

'''
Sunny = 0,0,1
Cool = 1,0,0
High = 1,0,0
True = 0,1
'''
case2 = [0,0,1,1,0,0,1,0,0,0,1]


features = ds1.columns.tolist()
features.remove("Play")
data_frame = pd.DataFrame([case1,case2], columns = features)
data_frame.head()

pred = dtc.predict(data_frame)
print("Prediction on whether play or not:\ncase1: {}\tcase2: {}".format(pred[0],pred[1]))

export_graphviz(dtc, out_file='tree_entropy.dot',
               feature_names=features,
               class_names=['do_not_play','play'], 
               filled=True)

# Convert to png
call(['dot', '-Tpng', 'tree_entropy.dot', '-o', 'tree_entropy.png', '-Gdpi=600'])

# Display in python
plt.figure(figsize = (14, 18))
plt.imshow(plt.imread('tree_entropy.png'))
plt.axis('off');
plt.show();

print(plot_tree(dtc,  class_names = ['do_not_play','play'], fontsize = 5, filled=True, impurity=True))