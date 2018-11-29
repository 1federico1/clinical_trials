from time import time
import numpy as np
from utils.datasets import heart_disease
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

x, y = heart_disease()
n = 3
randomize = np.random.permutation(len(x))
x = x[randomize]
y = y[randomize]
x = np.squeeze(x)
y = np.squeeze(y)

x_split = np.array_split(x, n + 1)
y_split = np.array_split(y, n + 1)

x_test_dataset = x_split[n]
y_test_dataset = y_split[n]


# x = MinMaxScaler().fit_transform(x)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
#
# estimator = KMeans()
# name = 'kmeans'
# labels = y
# data = x
# sample_size = len(y)
# t0 = time()
# estimator.fit(data)
#
# print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
#       % (name, (time() - t0), estimator.inertia_,
#          metrics.homogeneity_score(labels, estimator.labels_),
#          metrics.completeness_score(labels, estimator.labels_),
#          metrics.v_measure_score(labels, estimator.labels_),
#          metrics.adjusted_rand_score(labels, estimator.labels_),
#          metrics.adjusted_mutual_info_score(labels, estimator.labels_),
#          metrics.silhouette_score(data, estimator.labels_,
#                                   metric='euclidean',
#                                   sample_size=sample_size)))
#
# reduced_data = PCA(n_components=2).fit_transform(data)
# estimator = KMeans()
# name = 'kmeans + pca'
# t0 = time()
# estimator.fit(reduced_data)
# print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
#       % (name, (time() - t0), estimator.inertia_,
#          metrics.homogeneity_score(labels, estimator.labels_),
#          metrics.completeness_score(labels, estimator.labels_),
#          metrics.v_measure_score(labels, estimator.labels_),
#          metrics.adjusted_rand_score(labels, estimator.labels_),
#          metrics.adjusted_mutual_info_score(labels, estimator.labels_),
#          metrics.silhouette_score(reduced_data, estimator.labels_,
#                                   metric='euclidean',
#                                   sample_size=sample_size)))
#
# # Visualize the results on PCA-reduced data
#
#
# plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# # Plot the centroids as a white X
# centroids = estimator.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1],
#             marker='x', s=169, linewidths=3,
#             color='w', zorder=10)
# plt.title('K-means clustering on the heart dataset (PCA-reduced data)\n'
#           'Centroids are marked with white cross')
#
# plt.show()
#
