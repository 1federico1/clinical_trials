# Suppose we have a number of different hospitals that want to mine jointly their patient data for clinical research.
# They don't want to reveal their data to each other. We need a way to compute data mining algorithms on the union
# of their databases, without ever pooling or revealing their data

# For the most part we assume that the results of data mining algorithms is deemed secure (is it true?)

# Secure Multiparty Computation: a set of parties with private inputs wish to jointly compute some function on their
# inputs

# First of all, lets get a dataset and let's divide it horizontally

import random
import numpy as np

from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from utils import plot

k = 5

x, y = datasets.load_breast_cancer(return_X_y=True)
print(x.shape)
print(np.unique(y))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

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
# WRONG: There is _only_ one classifier, we need to train it on the _union_ of all the datasets
# MAYBE: each node has its own knn classifier trained with its private data (why they should not train classifiers
# with the information they have ?)

# shuffle array
randomize = np.random.permutation(len(x))
x = x[randomize]
y = y[randomize]
x = np.squeeze(x)
y = np.squeeze(y)
n = 3

x_split = np.array_split(x, n + 1)
y_split = np.array_split(y, n + 1)

local_knn_classifiers = []

# Just a test to compare classifications
for node_idx in range(n):
    x_part = x_split[node_idx]
    y_part = y_split[node_idx]

    x_part_train, x_part_test, y_part_train, y_part_test = train_test_split(x_part, y_part, test_size=0.25)

    knn_part = KNeighborsClassifier()
    knn_part.fit(x_part_train, y_part_train)

    y_part_pred = knn_part.predict(x_part_test)
    balanced_accuracy_score = metrics.balanced_accuracy_score(y_part_test, y_part_pred)
    print('Balanced accuracy score on local private dataset for node %d : %.2f' % (node_idx, balanced_accuracy_score))

# Nodes use classifiers trained on their complete data
for node_idx in range(n):
    knn_part = KNeighborsClassifier()
    knn_part.fit(x_split[node_idx], y_split[node_idx])
    local_knn_classifiers.append(knn_part)


# Divide the problem into two sub-problems
# 1. the k nearest neighbors of a query point could be distributed among the n nodes.
# So for a node to calculate its local classification of the query point it has to first determine which points
# in its database are among the k nearest neighbors of the query point.
#
# sketch of the algorithm
# 1. Given an instance x to be classified, each node computes the distance between x and each point y in its database
# , d(x,y), selects k-smallest distances (locally) and stores them in a local distance vector ldv

x_test_dataset = x_split[n]
y_test_dataset = y_split[n]

for local_knn in local_knn_classifiers:
    y_test_pred = local_knn.predict(x_test_dataset)
    print('Balanced accuracy score on test dataset for node %d : %.2f' % (
    local_knn_classifiers.index(local_knn), metrics.balanced_accuracy_score(y_test_pred, y_test_dataset)))

# n_neighbors should be equal to the size of the train test (return all the points)
# in this case i think it's sufficient to average it to 100
# Note that vectors are already sorted, so the topk of each node are the first k

y_pred = []

gdv = []
rounds = 10

p0 = 1.0
d = 0.75


def randomization(p0, d, r):
    return p0 * d ** (r - 1)


# select k nearest distances globally and store them in gdv

def knearest_global(node_idx_curr, node_idx_prev, round):
    pr = randomization(p0, d, round)
    curr_top_k = ldv[node_idx_curr][0].flatten()
    prev_top_k = ldv[node_idx_prev][0].flatten()
    stack = np.hstack((curr_top_k, prev_top_k))
    stack.sort()
    real_top_k = stack[:k]

    # compute sub-vector that contains the values of curr_top_k that contribute to the current topk vector
    curr_contribution = np.setdiff1d(real_top_k, prev_top_k)
    m = len(curr_contribution)

    if m == 0:
        # node_idx_curr does not have any values to contribute to the current topk.
        # In this case node_idx_curr simply passes on the global topk vector as its output, there's no
        # randomization needed because the node doesn't expose its own values
        return prev_top_k.tolist()
    else:
        if random.random() >= pr:
            return real_top_k.tolist()
        else:
            res = prev_top_k[0: k - m]
            # insert in the output m random values
            start = real_top_k[k - 1]
            end = prev_top_k[k - m]
            res = res.tolist()
            for _ in range(m):
                rand = np.random.uniform(start, end)
                res.append(rand)
            return res


gdv_i = None

for sample_id in range(len(y_test_dataset)):

    print('SAMPLE %d' % (sample_id))
    x_test_sample = x_test_dataset[sample_id].reshape(1, -1)
    y_test_sample = y_test_dataset[sample_id]
    ldv = []

    for local_knn in local_knn_classifiers:
        node_idx = local_knn_classifiers.index(local_knn)

        ldv_i = local_knn.kneighbors(x_test_sample, n_neighbors=k)
        ldv.append(ldv_i)

    # Now we need to find GLOBALLY the the k nearest distances and store them in gdv

    node_idx_curr = 0
    node_idx_prev = 0

    for r in range(rounds):
        print('Round: %d' % r)
        # alg init: curr = prev, then curr=1 and prev = 0
        node_idx_curr = 0
        node_idx_prev = 0
        gdv_i = knearest_global(node_idx_curr, node_idx_prev, r)
        gdv.append(gdv_i)
        print(gdv_i)
        for idx in range(n):
            node_idx_prev = idx
            node_idx_curr = idx + 1
            if node_idx_curr >= n:  # then the ring is closed
                break
            gdv_i = knearest_global(node_idx_curr, node_idx_prev, r)
            print(gdv_i, node_idx_prev, node_idx_curr)
        print()

    gdv = gdv_i

    # At the end of the rounds gdv = real_gdv (CHECK FOR PRIVACY BUGS: maybe with this implementation a node can snoop
    # other nodes' topk)

    real_gdv = []
    for ldv_i in ldv:
        dists = ldv_i[0].flatten()
        real_gdv.append(dists)

    real_gdv = np.array(real_gdv).flatten()
    real_gdv.sort()

    print('REAL TOP K DISTANCES ', real_gdv[:5])
    print('GLOBAL DISTANCE VECTOR ', gdv)
    # sanity check: to which database the top k belong ?
    for v in real_gdv[:5]:
        for node_idx in range(n):
            data = ldv[node_idx][0].flatten()
            if v in data:
                print(v, node_idx)

    # So now every node knows the k nearest distances

    # CLASSIFICATION
    # After each nodes determines the points in its database which are within the kth nearest distance from x, each
    # node computes a local classification vector of the query instance, where the ith element is the amount of
    # votes the ith class received from the points in this node's database which are among the k  nearest neighbors.
    # Note: so, i need to compute for every node the classification of x ?
    # The nodes then participate to find a global classification vector

    k_point = gdv[-1]

    lcv = []

    # lcv_i is a tuple -> lcv_i(class_index) += 1
    node_idx = 0
    print('CLASSIFICATION')
    for local_knn in local_knn_classifiers:

        lcv_i = [0, 0]

        ldv_i = local_knn.kneighbors(x_test_sample, n_neighbors=k)
        # from the paper isn't very clear. My interpretation: find all points that are in the radius of the k-th points
        dists = ldv_i[0].flatten()
        ids = ldv_i[1].flatten()

        dist_id = 0
        for dist in dists:
            if dist <= k_point: #dists is sorted actually TODO: break for when condition is false
                idx = ids[dist_id]
                cl = y_split[node_idx][idx]
                print(node_idx, idx, cl)
                lcv_i[cl] += 1
            else:
                break
            dist_id += 1

        prediction = local_knn.predict(x_test_sample)
        my_prediction = np.argmax(lcv_i)
        if prediction != my_prediction:
            prediction = local_knn.predict(x_test_sample)
        lcv.append(lcv_i)
        node_idx += 1

    # in this case we have that node 0 doesn't have any point in lcv. So he knows that the other nodes have all the
    # other distances. But i think this is part of the algorithm, since the global distances vector is public and
    # every node can compute it. If node 0 colludes with node 1, they will find out that all the classification is
    # made by node 2, that has the most classification power.
    #  Maybe with a better randomization (see previous _todo_) the problem will happen less,
    # i.e. every node has more or less the same power in deciding the classification of a test point.

    # let's say the random values are known only to a trusted third party
    random_values = np.random.randint(100, size=2)

    gcv = np.copy(random_values)

    # secure sum (this is a local simplification, off course. But the main idea is that every node adds to
    # the global vector he receives its class values. A node doesn't know in which position he is of the rings and
    # the local classification values of every other node. Still he can collude with the others.

    for lcv_i in lcv:
        node_idx = lcv.index(lcv_i)
        pos = 0
        for val in lcv_i:
            gcv[pos] += val
            pos += 1

    print('final gcv (what the final node sees) ', gcv)
    print('random values ', random_values)
    real_gcv = gcv - random_values
    print('real gcv ', real_gcv)

    cls = np.argmax(real_gcv)

    print('REAL CLASS: %d\n'
          'PREDICTED CLASS: %d' % (y_test_dataset[sample_id], cls))
    y_pred.append(cls)

print('EVALUATION')

plot.print_metrics(y_test_dataset, y_pred)
