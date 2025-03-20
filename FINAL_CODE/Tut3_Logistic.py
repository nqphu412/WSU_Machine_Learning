# -*- coding: utf-8 -*-
"""
Created 2023

@author: Geron, A. (2019). Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow
"""
import numpy as np

import matplotlib.pyplot as plt

#load data

from sklearn import datasets
iris = datasets.load_iris()  
list(iris.keys())

#['data','target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename']

X = iris["data"][:, 3:] # petal width
y = (iris["target"] == 2).astype(np.int)  #  1  if Iris-Virginica, else 0


######## training ######################

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)  

#predict: estimated probabilities for flowers with petal widths 
#varying from 0 to 3 cm (Figure 4-23):

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")
# + more Matplotlib code to make the image look pretty

log_reg.predict([[1.7], [1.5]])
#array([1, 0])

###################### training with two features #############

X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.int)

log_reg = LogisticRegression(solver="lbfgs", C=10**10, random_state=42)
log_reg.fit(X, y)

x0, x1 = np.meshgrid(
        np.linspace(2.9, 7, 500).reshape(-1, 1),
        np.linspace(0.8, 2.7, 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = log_reg.predict_proba(X_new)

plt.figure(figsize=(10, 4))
plt.plot(X[y==0, 0], X[y==0, 1], "bs")
plt.plot(X[y==1, 0], X[y==1, 1], "g^")

zz = y_proba[:, 1].reshape(x0.shape)
contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)


left_right = np.array([2.9, 7])
boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]

plt.clabel(contour, inline=1, fontsize=12)
plt.plot(left_right, boundary, "k--", linewidth=3)
plt.text(3.5, 1.5, "Not Iris virginica", fontsize=14, color="b", ha="center")
plt.text(6.5, 2.3, "Iris virginica", fontsize=14, color="g", ha="center")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.axis([2.9, 7, 0.8, 2.7])

plt.show()

########################softmax for multiclass classification###############

X = iris["data"][:, (2, 3)] # petal length, petal width
y = iris["target"] 

softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)

softmax_reg.fit(X, y)

softmax_reg.predict_proba([[5, 2]])

# array([[6.38014896e-07, 5.74929995e-02, 9.42506362e-01]])

softmax_reg.predict([[5, 2]])

#array([2])

x0, x1 = np.meshgrid(
        np.linspace(0, 8, 500).reshape(-1, 1),
        np.linspace(0, 3.5, 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]


y_proba = softmax_reg.predict_proba(X_new)
y_predict = softmax_reg.predict(X_new)

from sklearn.metrics import confusion_matrix
y_predict1=softmax_reg.predict(X)
confusion_matrix(y, y_predict1)

plt.figure(figsize=(10, 4))
plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris virginica")
plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris versicolor")
plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris setosa")
