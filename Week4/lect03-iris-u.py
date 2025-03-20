# -*- coding: utf-8 -*-
"""
Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow
Lect 3
"""
import numpy as np

import matplotlib.pyplot as plt

#load data

from sklearn import datasets

iris = datasets.load_iris( ) 

list(iris.keys())

#['data','target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename']

X = iris["data"][:, 3:] # petal width
y = (iris["target"] == 2).astype(np.int32)  #  1  if Iris-Virginica, else 0


#training

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



##################Confusion Matrix##############

y_pred = log_reg.predict(X)

from sklearn.metrics import confusion_matrix

print("y=\n",y,"\n y_pred=\n", y_pred)

print("comfusion matrix:", confusion_matrix(y,y_pred))

#############precision and recall################

from sklearn.metrics import precision_score, recall_score

print("precision score:", precision_score(y, y_pred))

print("recall score:", recall_score(y, y_pred))

########################softmax

X = iris["data"][:, (2, 3)] # petal length, petal width
y = iris["target"] 

softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)

softmax_reg.fit(X, y)

print("softmax predict proba for [5,2]", softmax_reg.predict_proba([[5, 2]]))

# array([[6.38014896e-07, 5.74929995e-02, 9.42506362e-01]])

print("softmax predict for [5,2]", softmax_reg.predict([[5, 2]]))

#array([2])


