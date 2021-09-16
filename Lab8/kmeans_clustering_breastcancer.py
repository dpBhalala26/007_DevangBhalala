# -*- coding: utf-8 -*-
"""KMeans_Clustering_BreastCancer.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1o9Q_SWpXuLApCT9Q3ev0gz4Xn16yMoWG

## Write a python program to perform K-Means clustering on Breast Cancer Data
"""

import numpy as np 
import pandas as pd 
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

dataset=datasets.load_breast_cancer()
dataset

print(dataset.data.shape)
print(dataset.target.shape)

print(dataset.feature_names)

print(dataset.target_names)

# 0 for benign and 1 for malignant

plt.scatter(dataset.data[:, 0], dataset.target, c='orange', marker='*')
plt.xlabel('Features')
plt.ylabel('Type of Cancer')
plt.show()

kmeans = KMeans(n_clusters=7, random_state=7)
prediction = kmeans.fit_predict(dataset.data)
print(prediction)

kmeans.cluster_centers_.shape
print(kmeans.cluster_centers_)

plt.scatter(dataset.data[:, 0], dataset.target, c='orange', marker='*')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='green', marker='+')
plt.title('Data points and cluster centroids')
plt.show()

from scipy.stats import mode
labels = np.zeros_like(prediction)
for i in range(10):
  mask = (prediction == i)
  labels[mask] = mode(dataset.target[mask])[0]

accuracy_score(dataset.target, labels)

import seaborn as sns

mat = confusion_matrix(dataset.target, labels)
ax = sns.heatmap(mat.T, square=True, annot=True, cbar=False, xticklabels=dataset.target_names, yticklabels=dataset.target_names)

plt.xlabel('true label')
plt.ylabel('predicted label')