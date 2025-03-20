#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning 
Module 08: Tree based methods 
"""

#%% Preamble: packages 
import numpy as np
from sklearn.datasets import load_iris

import graphviz 
# If you cannot import graphviz then you may need to install it: 
# pip install graphviz


import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import export_text
from sklearn.tree import export_graphviz, plot_tree

from sklearn.inspection import DecisionBoundaryDisplay

#%% ########### Section 1: use decision tree regression ###########
# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=1, linestyle='dashed')
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=1, linestyle='dotted')
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()



#%% ########### Section 2: use decision tree classification ###########   
# A teeny tiny toy example: logic XOR
X = [[0, 0], [1, 1], [0,1],[1,0]]
Y = [0, 0,1,1]
clf = DecisionTreeClassifier()
clf = clf.fit(X, Y)

# Make prediction
clf.predict([[2., 2.]])
clf.predict_proba([[2., 2.]])

#%% Visualise decision tree

iris = load_iris()
X, y = iris.data, iris.target
clf = DecisionTreeClassifier()
clf = clf.fit(X, y)
plt.figure(figsize=(36, 24))
plot_tree(clf)

dot_data = export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph

#%% Make it more colorful
dot_data = export_graphviz(clf, out_file=None, 
                     feature_names=iris.feature_names,  
                     class_names=iris.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph
# Save it to a pdf named iris.pdf: 
# graph.render("iris")

#%% Print actual decision
iris = load_iris()
decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
decision_tree = decision_tree.fit(iris.data, iris.target)
r = export_text(decision_tree, feature_names=iris['feature_names'])
print(r)

#%% Visualise decision boundaries
# Parameters
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02

plt.figure(figsize=(12, 8))

for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
    # We only take the two corresponding features
    X = iris.data[:, pair]
    y = iris.target

    # Train
    clf = DecisionTreeClassifier().fit(X, y)

    # Plot the decision boundary
    ax = plt.subplot(2, 3, pairidx + 1)
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        ax=ax,
        xlabel=iris.feature_names[pair[0]],
        ylabel=iris.feature_names[pair[1]],
    )

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(
            X[idx, 0],
            X[idx, 1],
            c=color,
            label=iris.target_names[i],
            cmap=plt.cm.RdYlBu,
            edgecolor="black",
            s=15,
        )

plt.suptitle("Decision boundaries of decision trees trained on pairs of features")
plt.legend(loc="lower right", borderpad=0, handletextpad=0)
_ = plt.axis("tight")


#%% Visualise decision boundaries
# Parameters
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02

plt.figure(figsize=(16, 12))

pair = [1,0]

X = iris.data[:, pair]
y = iris.target

# =============================================================================
# Your code goes from here
# ...
# ...
# 
# You code finishes here
# =============================================================================

