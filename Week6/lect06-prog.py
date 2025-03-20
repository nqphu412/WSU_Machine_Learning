# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import matplotlib.pyplot as plt

x1 = 2 * np.random.rand(100, 1) # generate linear random data
x2 = 4 + 4 * x1 + np.random.randn(100, 1)
x3 = x1**2 + x2**2

X = np.c_[x1, x2, x3]
			
X_centered = X - X.mean(axis=0)
U, s, V = np.linalg.svd(X_centered)


c1 = V[0]
c2 = V[1]

############ projection


W2=V[:2].T
X2D = X_centered.dot(W2)	


#########################

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X2D = pca.fit_transform(X)

print(pca.components_)

print(pca.explained_variance_ratio_)



##################### find the optimal d, 1st way
###############mnist example, d=0.95

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.target = mnist.target.astype(np.uint8)
from sklearn.model_selection import train_test_split

X = mnist["data"]
y = mnist["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)
pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
d
#154
pca = PCA(n_components=d)
X_reduced = pca.fit_transform(X_train)


################################################use n_components >0, <1
# second way 
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)
pca.n_components_
#154
np.sum(pca.explained_variance_ratio_)
#0.950373730459223

####################################plot d vs explained variance
##########3rd way

plt.figure(figsize=(6,4))
plt.plot(cumsum, linewidth=3)
plt.axis([0, 400, 0, 1])
plt.xlabel("Dimensions")
plt.ylabel("Explained Variance")
plt.plot([d, d], [0, 0.95], "k:")
plt.plot([0, d], [0.95, 0.95], "k:")
plt.plot(d, 0.95, "ko")
plt.annotate("Elbow", xy=(65, 0.85), xytext=(70, 0.7),
             arrowprops=dict(arrowstyle="->"), fontsize=16)
plt.grid(True)
#save_fig("explained_variance_plot")
plt.show()

from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    print(".", end="") # not shown 
    inc_pca.partial_fit(X_batch)

X_reduced = inc_pca.transform(X_train)



