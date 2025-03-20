#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning COMP3032
Module 09 - SVM

@author: yiguo
"""
#%% Preamble: packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC,SVC,SVR
from sklearn.inspection import DecisionBoundaryDisplay


#%% ########### Section 1: SVM classification demo ###########
np.random.seed(1)

X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=1.1)

plt.figure(figsize=(5, 5))
C=10
# "hinge" is the standard SVM loss
clf = SVC(C=C, kernel='linear',random_state=42).fit(X, y)
# obtain the support vectors through the decision function
decision_function = clf.decision_function(X)
# we can also calculate the decision function manually
# decision_function = np.dot(X, clf.coef_[0]) + clf.intercept_[0]
# The support vectors are the samples that lie within the margin
# boundaries, whose size is conventionally constrained to 1
support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
support_vectors = X[support_vector_indices]

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
    clf,X,ax=ax,grid_resolution=50,plot_method="contour",colors="k",
    levels=[-1, 0, 1],alpha=0.5,linestyles=["--", "-", "--"])
plt.scatter(support_vectors[:, 0],support_vectors[:, 1],s=100,linewidth=1,
    facecolors="none",edgecolors="k")
plt.title("C=" + str(C))
plt.tight_layout()
plt.show()

# Q1: Try out different values of C and plot the decision boundaries. 
# Q2: Try nonlinear SVM and varies C values to see the differnces. 

#%% ########### Section 2: SVM regressino demo ###########

# =============================================================================
# # Generate sample data, case 1: linear function
# =============================================================================
np.random.seed(1)
n=20
X = np.sort(5 * np.random.rand(n, 1), axis=0)
y = (2*X + 1).ravel()
truey = y.copy()
# add noise to targets
# y[::5] += 5 * (0.5 - np.random.rand(8))

y += np.random.normal(0,1,n)
y[5] = y[5]+10

# Fit regression model
svr_lin = SVR(kernel="linear", C=1)
# Choose epsilon value will change the number of the SVs. 
svr_lin = SVR(kernel="linear", C=1, epsilon=1.1)

# Look at the results
plt.figure(figsize=(10, 10))
plt.plot(X,truey,color='g',lw=4,label="True model")
plt.scatter(X,y, c=y/y*1, s=20, cmap=plt.cm.Paired,label="Training data")
plt.plot(X,svr_lin.fit(X, y).predict(X),color='r',lw=2,label="Predicted model",)
plt.scatter(X[svr_lin.support_],y[svr_lin.support_],
    facecolor="none",edgecolor='k',s=150,
    label="support vectors",)
plt.legend(loc="upper left",ncol=1,fancybox=True,shadow=True,)

plt.title("Support Vector Regression", fontsize=14)
plt.show()
#%%
# =============================================================================
# # Generate sample data, case 1: nonlinear function
# =============================================================================

X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()
# y = (2*X + 1).ravel()
truey = y.copy()

# add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))

# Fit regression model

svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
svr_lin = SVR(kernel="linear", C=100, gamma="auto")
svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)

# Look at the results

lw = 2

svrs = [svr_rbf, svr_lin, svr_poly]
kernel_label = ["RBF", "Linear", "Polynomial"]
model_color = ["r", "m", "b"]

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 10), sharey=True)
for ix, svr in enumerate(svrs):
    axes[ix].plot(X,truey,color='g',lw=4,label="True model")
    axes[ix].scatter(X,y, c=y/y*1, s=20, cmap=plt.cm.Paired,label="Training data")
    axes[ix].plot(
        X,
        svr.fit(X, y).predict(X),
        color=model_color[ix],
        lw=lw,
        label="{} model".format(kernel_label[ix]),
    )
    axes[ix].scatter(
        X[svr.support_],
        y[svr.support_],
        facecolor="none",
        edgecolor=model_color[ix],
        s=150,
        label="{} support vectors".format(kernel_label[ix]),
    )
    axes[ix].scatter(
        X[np.setdiff1d(np.arange(len(X)), svr.support_)],
        y[np.setdiff1d(np.arange(len(X)), svr.support_)],
        facecolor="none",
        edgecolor="k",
        s=50,
        label="other training data",
    )
    axes[ix].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=1,
        fancybox=True,
        shadow=True,
    )

fig.text(0.5, 0.092, "Input", ha="center", va="center")
fig.text(0.1, 0.5, "Target", ha="center", va="center", rotation="vertical")
fig.suptitle("Support Vector Regression", fontsize=14)
plt.show()

# Q1: Try out different values of C and plot the results. 
# Q2: For the linear SVR case, it seems to be too many support vectors. 
# Why? How to reduce them? 
# Q3: Which model is the best?
 
#%% ########### Section 3: task to do ###########
# =============================================================================
# Task: use SVM to classify handwritten digits: 
# a. Train a linear SVM on all training data and apply trained model to the test set. 
# b. Plot 5 support vectors (images). How do they look like? 
# c. Use the training data to create training and validation sets. Use the 
# validation set to find the best values for the hyperparameters of a type of 
# nonlinear SVM (your choice). Use the optimal hyperparameters values, compute 
# the error on the test set. How does it compare with the results from KNN?
# =============================================================================

import numpy as np
import pandas as pd

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

#---  Collecting all the imports used in this tutorial here: 

# metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# other
import timeit
import tracemalloc
# Make sure the files are in the same directory as the source code.
mtrain = pd.read_csv("mnist_train.csv", header=0)
mtest = pd.read_csv("mnist_test.csv", header=0)

print('size of training data: ', mtrain.shape)
print('size of test data: ', mtest.shape)

# Prepare the data. Just run once
# Let's just pick 0, 3 and 8 
digits = [0,3,8]
traindata = mtrain.loc[mtrain['label'].isin(digits)]
testdata = mtest.loc[mtest['label'].isin(digits)]
# set up the data: 
X = traindata.drop(['label'], axis=1)
y = traindata['label'].astype(np.float64)   

# set up test data:
X_test = testdata.drop(['label'], axis=1)
y_test = testdata['label'].astype(np.float64)

# Take the first image and plot it
idx = 20 # You can change it to see others
img = np.c_[X.iloc[idx,:]].reshape(28,28)
plt.imshow(img, cmap="Greys")
plt.title(f'sample label is {y.iloc[idx].astype(np.int64)}')
plt.show()
#%% Section 3: (a) 
# =============================================================================
# (a) Train a linear SVM on all training data and apply trained model to the test set. 
# =============================================================================
# Note: tracemalloc is used here to check memory usage
# Time to run the process:  8.46361431600053
# Current memory usage is 1045.987093MB; Peak was 1799.118474MB

# =============================================================================
# Your code goes from here
# ...
# ...
# 
# You code finishes here
# =============================================================================

# Assume your model produce the prediction in y_pred:
# classification accuracy
confusion_matrix(y_pred, y_test)
print(accuracy_score(y_test, y_pred)*100)

#%% Section 3: (b) 
# =============================================================================
# (b) Plot 5 misclassified images and 5 support vectors (images). How do they look like? 
# =============================================================================
indMisclassification = np.where((y_test == y_pred) == False)
indmis = indMisclassification[0]

# =============================================================================
# Your code goes from here
# ...
# ...
# 
# You code finishes here
# =============================================================================

# Hint: use the code in previous section to get support vectors of the trained
# linear SVM. 

#%% Section 3: (c) 
# =============================================================================
# Use the training data to create training and test sets. Use the training 
# set to find the best values for the hyperparameters of a type of nonlinear SVM 
# (your choice). Use the optimal hyperparameters values, compute the error on the 
# test set. How does it compare with the results from KNN?
# =============================================================================
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

# =============================================================================
# Cross validation on large data set can be very slow. Therefore, for demo 
# purpose, we only take a small portion of the training data to do it. 
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, random_state=42)

# Use X_train and y_train to do CV.
# =============================================================================
# Your code goes from here
# ...
# ...
# 
# You code finishes here
# =============================================================================

# Hint: You can generate a list of values for each hypeparameter and define
# a list of models using different combinations of hyperparameter values. Then 
# CV across all these models. You can use functions in model_selection in sklearn 
# to do it. 

# Assume you have chosen the best model stored in bestmodel. Test it out!
bestmodel.score(X_test, y_test)
