import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat

# 减少维度至一维
data = loadmat('./ex7data1.mat')
X = data['X']

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(X[:, 0], X[:, 1])
plt.show()


def pca(X):
    X = (X - X.mean()) / X.std()

    X = np.matrix(X)
    cov = (X.T * X) / X.shape[0]

    U, S, V = np.linalg.svd(cov)

    return U, S, V


def project_data(X, U, k):
    U_reduced = U[:, :k]
    return np.dot(X, U_reduced)


def recover_data(Z, U, k):
    U_reduced = U[:, :k]
    return np.dot(Z, U_reduced.T)


U, S, V = pca(X)
Z = project_data(X, U, 1)
X_recovered = recover_data(Z, U, 1)
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(list(X_recovered[:, 0]), list(X_recovered[:, 1]))
plt.show()

# 图像处理
faces = loadmat('./ex7faces.mat')
X = faces['X']


def plot_n_image(X, n):
    """ plot first n images
    n has to be a square number
    """
    pic_size = int(np.sqrt(X.shape[1]))
    grid_size = int(np.sqrt(n))

    first_n_images = X[:n, :]

    fig, ax_array = plt.subplots(nrows=grid_size, ncols=grid_size,
                                 sharey=True, sharex=True, figsize=(8, 8))

    for r in range(grid_size):
        for c in range(grid_size):
            ax_array[r, c].imshow(first_n_images[grid_size * r + c].reshape((pic_size, pic_size)))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))



face = np.reshape(X[3,:], (32, 32))
U,S,V=pca(X)
Z=project_data(X,U,100)

X_recovered=recover_data(Z,U,100)
face = np.reshape(X_recovered[3,:], (32, 32))
plt.imshow(face)
plt.show()