# Suppose we have a number of different hospitals that want to mine jointly their patient data for clinical research.
# They don't want to reveal their data to each other. We need a way to compute data mining algorithms on the union
# of their databases, without ever pooling or revealing their data

# For the most part we assume that the results of data mining algorithms is deemed secure (is it true?)

# Secure Multiparty Computation: a set of parties with private inputs wish to jointly compute some function on their
# inputs

# First of all, lets get a dataset and let's divide it horizontally
from typing import List

from numpy.core.multiarray import ndarray
from sklearn import datasets, metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from utils import plot
import numpy as np
import random

random_state = 42

x, y = datasets.load_breast_cancer(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=random_state)

complete_data_knn = KNeighborsClassifier()
complete_data_knn.fit(x_train, y_train)
y_pred = complete_data_knn.predict(x_test)
balanced_accuracy_score = metrics.balanced_accuracy_score(y_test, y_pred)
print('Balanced accuracy score on non private dataset: %.2f' % (balanced_accuracy_score))
# plot.print_metrics(y_test, y_pred)

test = x_test[1]
test = test.reshape(1, -1)
print(complete_data_knn.kneighbors(test))  # array([distances], [indexes of k-nearest neighbors]). k = 5

# We now divide the dataset in n parts, to simulate n local parts
# Actually we split in n+1 parts so that the last one will be used as test

n = 3
x_split: List[ndarray] = np.array_split(x, n + 1)
y_split: List[ndarray] = np.array_split(y, n + 1)

local_knn_classifiers = []

for node_idx in range(n):
    x_part = x_split[node_idx]
    y_part = y_split[node_idx]

    x_part_train, x_part_test, y_part_train, y_part_test = train_test_split(x_part, y_part, test_size=0.25)

    knn_part = KNeighborsClassifier()
    knn_part.fit(x_part_train, y_part_train)

    y_part_pred = knn_part.predict(x_part_test)
    balanced_accuracy_score = metrics.balanced_accuracy_score(y_part_test, y_part_pred)
    print('Balanced accuracy score on local private dataset for part %d : %.2f' % (node_idx, balanced_accuracy_score))

    local_knn_classifiers.append(knn_part)

    y_part_pred_diff_dataset = knn_part.predict(x_split[n])
    balanced_accuracy_score_test = metrics.balanced_accuracy_score(y_part_pred_diff_dataset, y_split[n])
    print('Balanced accuracy score on test data for part %d : %.2f' % (node_idx, balanced_accuracy_score_test))
    print()



# K nearest neighbors in a private distributed setting
# Each node calculates k smallest distances between x and the points in their database (locally)
# and then we can use a privacy preserving algorithm to determine the k smallest distances between x and the points
# in the union of the databases or kth nearest distance (globally).
# Use multi-round topk algorithm to determine the k smallest distances before determining the k-th smallest distance

# sketch of the algorithm
# 1. Given an instance x to be classified, each node computes the distance between x and each point y in its database
# , d(x,y), selects k-smallest distances (locally) and stores them in a local distance vector ldv

x_test_dataset = x_split[n]
y_test_dataset = y_split[n]

ldv = []

x_test_sample = x_test_dataset[0].reshape(1, -1)
y_test_sample = y_test_dataset[0]

for node_knn in local_knn_classifiers:
    topk = node_knn.kneighbors(x_test_sample) #possible problem: the indexes refer to the local datasets
    ldv.append(topk)



# 2. using ldv as inputs, the nodes use the adapted privacy preserving topk selection protocol to select k
# nearest distances (globally), and stores them in global distance vector gdv.

# implementation of top k queries

