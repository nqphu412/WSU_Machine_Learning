# -*- coding: utf-8 -*-

#Geron, A. (2019). Hands-on Machine Learning with Scikit-Learn, Keras and TensorFlow
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#import os




################

from sklearn.datasets import make_blobs
blob_centers = np.array(
    [[ 0.2,  2.3],
     [-1.5 ,  2.3],
     [-2.8,  1.8],
     [-2.8,  2.8],
     [-2.8,  1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
X, y = make_blobs(n_samples=2000, centers=blob_centers,
                  cluster_std=blob_std, random_state=7)
#Now let's plot them:

def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)
plt.figure(figsize=(8, 4))
plot_clusters(X)
# save_fig("blobs_plot")
plt.show()




################# train and predict

from sklearn.cluster import KMeans
k = 5
kmeans = KMeans(n_clusters=k)
y_pred = kmeans.fit_predict(X)

###############################################
y_pred is kmeans.labels_
centers=kmeans.cluster_centers_
kmeans.labels_

print("centroids=", centers,"\n")
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
y_pred1=kmeans.predict(X_new)
#array([0, 0, 3, 3], dtype=int32)

print("X_new is assigned to", y_pred1,"\n")

y_preddist=kmeans.transform(X_new)

print("X_new distances", y_preddist,"\n")

######################
from sklearn.metrics import silhouette_score
silhouette_score(X, kmeans.labels_)

print("the inertia ", kmeans.inertia_,"\n")

print("the score ", kmeans.score(X),"\n")

#################

from sklearn.cluster import MiniBatchKMeans

minibatch_kmeans = MiniBatchKMeans(n_clusters=5)
minibatch_kmeans.fit(X)
y_pred2=minibatch_kmeans.predict(X_new)
silhouette_score(X, minibatch_kmeans.labels_)


########################### for preprocessing


from sklearn.datasets import load_digits
X_digits, y_digits = load_digits(return_X_y=True)
#split it into a training set and a test set:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)

#fit a Logistic Regression model and evaluate it on the test set:

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(X_train, y_train)  #one-vs-rest
LogisticRegression(max_iter=5000, multi_class='ovr', random_state=42)
log_reg_score = log_reg.score(X_test, y_test)
print("logistic reg score= ", log_reg_score)

#0.9688888888888889

#create a pipeline 
#first cluster the training set into 50 clusters and replace the images with their distances to the 50 clusters
#then apply a logistic regression model:

from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ("kmeans", KMeans(n_clusters=50, random_state=42)),
    ("log_reg", LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)),
])
pipeline.fit(X_train, y_train)

pipeline_score = pipeline.score(X_test, y_test)
print("pineline score=", pipeline_score)
#0.98


