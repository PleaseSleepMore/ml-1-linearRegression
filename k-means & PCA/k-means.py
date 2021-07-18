import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat


def find_closest_centroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m)

    for i in range(m):
        min_dist = 1000000
        for j in range(k):
            dist = np.sum((X[i, :] - centroids[j, :]) ** 2)
            if dist < min_dist:
                min_dist = dist
                idx[i] = j
    return idx


data = loadmat("./ex7data2.mat")
X = data['X']
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
idx = find_closest_centroids(X, initial_centroids)
print(idx[0:3])

data2 = pd.DataFrame(data.get('X'), columns=['X1', 'X2'])
sb.set(context='notebook', style='white')
sb.lmplot('X1', 'X2', data=data2, fit_reg=False)
plt.show()


def compute_center(X, idx, k):
    m, n = X.shape;
    center = np.zeros((k, n))

    for i in range(k):
        indics = np.where(idx == i)
        center[i, :] = (np.sum(X[indics, :], axis=1) / len(indics[0])).ravel()

    return center


print(compute_center(X, idx, 3))

# 多次迭代
def run_kmeans(X, initial_center, max_iters):
    m, n = X.shape
    k = initial_center.shape[0]
    idx = np.zeros(m)
    center = initial_center

    for i in range(max_iters):
        idx = find_closest_centroids(X, center)
        center = compute_center(X, idx, k)

    idx = find_closest_centroids(X, center)
    return idx, center


# 随机初始化
def init_centroids(X, k):
    m, n = X.shape
    centroids = np.zeros((k, n))
    idx = np.random.randint(0, m, k)

    for i in range(k):
        centroids[i, :] = X[idx[i], :]

    return centroids

print(init_centroids(X,3))


# actual_runing and visualization
idx, centroids = run_kmeans(X, init_centroids(X,3), 10)
cluster1 = X[np.where(idx == 0)[0], :]
cluster2 = X[np.where(idx == 1)[0], :]
cluster3 = X[np.where(idx == 2)[0], :]

fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(cluster1[:, 0], cluster1[:, 1], s=30, color='r', label='Cluster 1')
ax.scatter(cluster2[:, 0], cluster2[:, 1], s=30, color='g', label='Cluster 2')
ax.scatter(cluster3[:, 0], cluster3[:, 1], s=30, color='b', label='Cluster 3')
ax.legend()
plt.show()

# 处理图像
from IPython.display import Image
image_data=loadmat('./bird_small.mat')
A=image_data['A']

# reshape the array
A=A/255
X=np.reshape(A,(A.shape[0]*A.shape[1],A.shape[2]))

# randomly initilize the centroids
initial_centroids=init_centroids(X,16)
idx,centroids=run_kmeans(X,initial_centroids,10)
idx=find_closest_centroids(X,centroids)

X_recovered=centroids[idx.astype(int),:]


X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))
plt.imshow(X_recovered)
plt.show()
# 可以看见图像进行了压缩，但图像的主要特征仍然存在



