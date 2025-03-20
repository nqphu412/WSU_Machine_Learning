#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning 
Module 07: nonparametric methods 
"""

#%% Preamble: packages 
import numpy as np
import numpy as np
import os
import pandas as pd

from sklearn import datasets

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

#%% ########### Section 1: KNN classification ###########
# to make the code output stable across runs
np.random.seed(42)

# To plot pretty figures
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# kNN classification

# set up iris data
# use sepal width, petal width to predict species
iris = datasets.load_iris()
X = iris["data"][:, (1, 3)]  # sepal width, petal width
y = iris["target"]

def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "b*", label="Setosa")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^", label="Versicolor")
    plt.plot(X[:, 0][y==2], X[:, 1][y==2], "ro", label="Virginica")

    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel("Sepal width", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="center right", fontsize=14)

# Just plot the data    
plt.figure(figsize=(7,6))
plot_dataset(X, y, [1.9, 4.5, 0.0, 2.6])
plt.show()

# Plot the decision boundary (class regions) 
def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)

k = 1
knn_clf = KNeighborsClassifier(n_neighbors=k).fit(X, y)

plt.figure(figsize=(7, 6))
plot_predictions(knn_clf, [1.9, 4.5, 0.0, 2.6])
plot_dataset(X, y, [1.9, 4.5, 0.0, 2.6])
plt.title("{} Neighbour".format(k),  fontsize=14)
plt.show()


#%% ########### Section 2: KNN regression (kernel smoothing) ###########
np.random.seed(0)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
T = np.linspace(0, 5, 500)[:, np.newaxis]
y = np.sin(X).ravel()

# Add noise to targets
y[::5] += 1 * (0.5 - np.random.rand(8))

n_neighbors = 5

for i, weights in enumerate(["uniform", "distance"]):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    y_ = knn.fit(X, y).predict(T)

    plt.subplot(2, 1, i + 1)
    plt.scatter(X, y, color="darkorange", label="data")
    plt.plot(T, y_, color="navy", label="prediction")
    plt.axis("tight")
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors, weights))

plt.tight_layout()
plt.show()

#%% ########### Section 3: Use KNN to predict hand written digits ###########
# Common imports
import numpy as np
import os
import pandas as pd

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

#---  Collecting all the imports used in this tutorial here: 
# kNN stuff
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

# PCA
from sklearn.decomposition import PCA

# preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

# other
import timeit
import tracemalloc

mtrain = pd.read_csv("mnist_train.csv", header=0)
mtest = pd.read_csv("mnist_test.csv", header=0)

print('size of training data: ', mtrain.shape)
print('size of test data: ', mtest.shape)

digit1 = 3
digit2 = 8
mtrain2 = mtrain.loc[(mtrain['label'] == digit1) | (mtrain['label'] == digit2)]
mtest2 = mtest.loc[ (mtest['label'] == digit1) | (mtest['label'] == digit2) ]

# set up the data: 
X = mtrain2.drop(mtrain2.columns[0], axis=1)
y = (mtrain2['label'] == digit2).astype(np.float64)   

# Take the first image and plot it
idx = 0 # You can change it to see others
img = np.c_[X.iloc[idx,:]].reshape(28,28)
plt.imshow(img, cmap="Greys")
plt.title(f'sample label is {y.iloc[idx].astype(np.int64)}')
plt.show()
#%% Section 3: (a) 
# =============================================================================
# (a) Create subsets of 2 digits from both the training and test data 
# (pick two digits, e.g. 3 and 8). Classify the test data using the training set, 
# with k = 3.
# =============================================================================
# Let's just pick "3" and "8" 
digit1 = 3
digit2 = 8
mtrain2 = mtrain.loc[(mtrain['label'] == digit1) | (mtrain['label'] == digit2)]
mtest2 = mtest.loc[ (mtest['label'] == digit1) | (mtest['label'] == digit2) ]

# set up the data: 
X = mtrain2.drop(mtrain2.columns[0], axis=1)
y = (mtrain2['label'] == digit2).astype(np.float64)   

# set up test data:
X_test = mtest2.drop(mtest2.columns[0], axis=1)
y_test = (mtest2['label'] == digit2).astype(np.float64)

# =============================================================================
# Your code goes from here
# ...
# ...
# Your code ends here
# =============================================================================

# y_pred is supposed to be the model prediction on test data. 
# classification accuracy
confusion_matrix(y_pred, y_test)
print(accuracy_score(y_test, y_pred)*100)

#%% Section 3: (b) 
# =============================================================================
# (b) Pick a misclassified example. How do the nearest neighbours look like?
# =============================================================================

# =============================================================================
# Your code goes from here
# ...
# ...
# Your code ends here
# =============================================================================

# Hint: use the plotting code shown at the begining to draw. 


#%% Section 3: (c) 
# =============================================================================
# Use the training data to create training and validation sets. Use the validation 
# set to find the best value for k. Use the k you selected, compute the error on 
# the test set. How does it compare with results from part (a)? 
# =============================================================================
X_train, X_validate, y_train, y_validate = train_test_split(X, y, 
                                                            test_size=0.5, 
                                                            random_state=42)

k_values = np.arange(1, 12)
results_acc = []

# =============================================================================
# Your code goes from here
# ...
# ...
# Your code ends here
# =============================================================================

# Hint: results_acc contains the accuracies for different k values in k_values. 


plt.figure
plt.plot(k_values, results_acc, '-*')
plt.show()

# Q1: What value of k is the best? 
# Q2: Is it cross validation? 

#%% Section 3: (d) 
# =============================================================================
# Apply the selected k to classify all digits of the test set. 
# =============================================================================
# Classify all digits of the test set

# =============================================================================
# Your code goes from here
# ...
# ...
# Your code ends here
# =============================================================================

# Hint: Just rearrange the data and apply KNN withe k value you choose from 
# part (c). 
# Q: is it the right way to select k? 
