# Suppose we have a number of different hospitals that want to mine jointly their patient data for clinical research.
# They don't want to reveal their data to each other. We need a way to compute data mining algorithms on the union
# of their databases, without ever pooling or revealing their data

# For the most part we assume that the results of data mining algorithms is deemed secure (is it true?)

# Secure Multiparty Computation: a set of parties with private inputs wish to jointly compute some function on their
# inputs

# First of all, let's get a dataset and let's divide it horizontally
from collections import defaultdict
import numpy as np
import random
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from utils import plot
from utils.datasets import heart_disease, skin_noskin, abalone
from utils.grid_search import grid_search, KNN_PARAMS
from utils.plot import print_metrics
from scipy.spatial import distance
from matplotlib import pyplot as plt
import pickle


def randomization(p0, d, r):
    return p0 * (d ** (r - 1))


def knearest_global_knn(node_idx_curr, node_idx_prev, round, ldv, gdv, p0=1.0, d=0.75, k=5):
    pr = randomization(p0, d, round)
    if DEBUG:
        print('Probability', pr)
    curr_top_k = ldv[node_idx_curr].flatten()

    gdv_prev = gdv[node_idx_prev]

    stack = np.hstack((curr_top_k, gdv_prev))
    # stack[::-1].sort()
    stack.sort()
    curr_top_k = stack[:k]

    # compute sub-vector that contains the values of curr_top_k that contribute to the current topk vector
    # curr_contribution = np.setdiff1d(curr_top_k, prev_top_k)
    curr_contribution = np.setdiff1d(curr_top_k, gdv_prev)
    m = len(curr_contribution)

    if m == 0:
        # node_idx_curr does not have any values to contribute to the current topk.
        # In this case node_idx_curr simply passes on the global topk vector as its output, there's no
        # randomization needed because the node doesn't expose its own values
        # return prev_top_k.tolist()
        return gdv_prev
    else:
        if random.random() > pr:
            return curr_top_k.tolist()
        else:
            # res = prev_top_k[0: k - m]
            res = gdv_prev[0: k - m]
            # insert in the output m random values
            start = curr_top_k[k - 1]  # last item
            # end = prev_top_k[k - m]
            end = gdv_prev[k - m]  # k - m th item
            # res = res.tolist()
            for _ in range(m):
                rand = np.random.uniform(low=start, high=end)
                res.append(rand)
            return sorted(res)


def knearest_global(node_idx_curr, node_idx_prev, round, ldv, gdv, p0=1.0, d=0.75):
    pr = randomization(p0, d, round)
    if DEBUG:
        print('Probability', pr)
    curr_top_k = ldv[node_idx_curr][0][0].flatten()
    prev_top_k = ldv[node_idx_prev][0][0].flatten()

    gdv_prev = gdv[node_idx_prev]

    # stack = np.hstack((curr_top_k, prev_top_k))
    stack = np.hstack((curr_top_k, gdv_prev))
    stack.sort()
    curr_top_k = stack[:k]

    # compute sub-vector that contains the values of curr_top_k that contribute to the current topk vector
    # curr_contribution = np.setdiff1d(curr_top_k, prev_top_k)
    curr_contribution = np.setdiff1d(curr_top_k, gdv_prev)
    m = len(curr_contribution)

    if m == 0:
        # node_idx_curr does not have any values to contribute to the current topk.
        # In this case node_idx_curr simply passes on the global topk vector as its output, there's no
        # randomization needed because the node doesn't expose its own values
        # return prev_top_k.tolist()
        return gdv_prev
    else:
        if random.random() < pr:
            return curr_top_k.tolist()
        else:
            # res = prev_top_k[0: k - m]
            res = gdv_prev[0: k - m]
            # insert in the output m random values
            start = curr_top_k[k - 1]  # last item
            # end = prev_top_k[k - m]
            end = gdv_prev[k - m]  # k - m th item
            # res = res.tolist()
            for _ in range(m):
                rand = np.random.uniform(start, end)
                res.append(rand)
            return sorted(res)


k = 5

# x, y = datasets.load_breast_cancer(return_X_y=True)
x, y = heart_disease()
# x, y = skin_noskin()
# x = MinMaxScaler().fit_transform(x)

# gs, x_train, y_train, x_test, y_test = grid_search(x, y, test_size=0.25, clf=KNeighborsClassifier(), params=KNN_PARAMS,
#                                                    cv=10, verbose=0, standardize=False,
#                                                    random_state=int(random.random()) * 10 ** 32)
# print(gs.best_params_, gs.best_score_)

print(x.shape)
print(np.unique(y))

# complete_data_knn = gs.best_estimator_
complete_data_knn = KNeighborsClassifier(n_neighbors=k)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
complete_data_knn.fit(x_train, y_train)
y_pred_baseline = complete_data_knn.predict(x_test)
baseline = metrics.balanced_accuracy_score(y_test, y_pred_baseline)


# plot.print_metrics(y_test, y_pred_baseline)


# We now divide the dataset in n parts, to simulate n local parts
# Actually we split in n+1 parts so that the last one will be used as test
# WRONG: There is _only_ one classifier, we need to train it on the _union_ of all the datasets
# MAYBE: each node has its own knn classifier trained with its private data (why they should not train classifiers
# with the information they have ?)

# shuffle array
# randomize = np.random.permutation(len(x))
# x = x[randomize]
# y = y[randomize]
# x = np.squeeze(x)
# y = np.squeeze(y)
# n = 20
#
# x_split = np.array_split(x, n + 1)
# y_split = np.array_split(y, n + 1)
#
# local_knn_classifiers = []

# Just a test to compare classifications
# for node_idx in range(n):
#     x_part = x_split[node_idx]
#     y_part = y_split[node_idx]
#
#     x_part_train, x_part_test, y_part_train, y_part_test = train_test_split(x_part, y_part, test_size=0.25)
#
#     knn_part = KNeighborsClassifier()
#     knn_part.fit(x_part_train, y_part_train)
#
#     y_part_pred = knn_part.predict(x_part_test)
#     baseline = metrics.balanced_accuracy_score(y_part_test, y_part_pred)
#     print('Balanced accuracy score on local private dataset for node %d : %.2f' % (node_idx, baseline))

# Nodes use classifiers trained on their complete data

# for node_idx in range(n):
#     print('Grid search node', node_idx)
#     # knn_part = KNeighborsClassifier(**gs.best_params_)
#     knn_part = KNeighborsClassifier()
#     knn_part.fit(x_split[node_idx], y_split[node_idx])
#     print('LOCAL KNN SCORE', knn_part.score(x_split[n], y_split[n]))
#     local_knn_classifiers.append(knn_part)

# Divide the problem into two sub-problems
# 1. the k nearest neighbors of a query point could be distributed among the n nodes.
# So for a node to calculate its local classification of the query point it has to first determine which points
# in its database are among the k nearest neighbors of the query point.
#
# sketch of the algorithm
# 1. Given an instance x to be classified, each node computes the distance between x and each point y in its database
# , d(x,y), selects k-smallest distances (locally) and stores them in a local distance vector ldv

# x_test_dataset = x_split[n]
# y_test_dataset = y_split[n]
#
# for node_idx in range(n):
#     local_knn = local_knn_classifiers[node_idx]
#     y_test_pred = local_knn.predict(x_test_dataset)
#     print('Balanced accuracy score on test dataset for node %d : %.2f' % (
#         node_idx, metrics.balanced_accuracy_score(y_test_pred, y_test_dataset)))


# n_neighbors should be equal to the size of the train test (return all the points)
# in this case i think it's sufficient to average it to 100
# Note that vectors are already sorted, so the topk of each node are the first k


def test_knn(p0=1.0, d=0.75, rounds=10, k=5, n=3):
    '''

    :param p0:
    :param d:
    :param rounds:
    :param k:
    :param n:
    :return:
    '''
    # data preparation

    x, y = heart_disease()
    # x, y = datasets.load_breast_cancer(return_X_y=True)
    # x, y = abalone()
    # x = MinMaxScaler().fit_transform(x)
    randomize = np.random.permutation(len(x))
    x = x[randomize]
    y = y[randomize]
    x = np.squeeze(x)
    y = np.squeeze(y)

    x_split = np.array_split(x, n + 1)
    y_split = np.array_split(y, n + 1)

    x_test_dataset = x_split[n]
    y_test_dataset = y_split[n]

    true_knn = KNeighborsClassifier(n_neighbors=k)
    true_x = []
    true_y = []
    for node_idx in range(n):
        local_data = x_split[node_idx]
        for value in local_data:
            true_x.append(value)
    for node_idx in range(n):
        local_data = y_split[node_idx]
        for label in local_data:
            true_y.append(label)

    true_knn.fit(true_x, true_y)
    true_pred = true_knn.predict(x_test_dataset)
    true_baseline = metrics.balanced_accuracy_score(y_test_dataset, true_pred)
    local_classifiers = []
    for node_idx in range(n):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_split[node_idx], y_split[node_idx])
        local_classifiers.append(knn)

    y_pred = []

    for sample_id in range(len(y_test_dataset)):

        if DEBUG:
            print('SAMPLE %d' % (sample_id))

        x_test_sample = x_test_dataset[sample_id].reshape(1, -1)
        ldv = []

        for node_idx in range(n):
            node_knn = local_classifiers[node_idx]
            ldv_i = node_knn.kneighbors(x_test_sample, n_neighbors=k)[0]
            ldv.append(ldv_i)

        gdv = [[np.random.uniform(10, 50) for _ in range(k)]]

        for r in range(rounds):

            # keep only the last item of gdv at next round
            if r > 0:
                gdv = [gdv[-1]]

            if DEBUG:
                print('Round: %d' % r)

            # alg init: curr = prev, then curr=1 and prev = 0
            node_idx_curr = 0
            node_idx_prev = 0
            gdv_i = knearest_global_knn(node_idx_curr, node_idx_prev, r, ldv, gdv, p0, d, k=k)
            gdv.append(gdv_i)
            gdv = [gdv[1]]  # remove randomly generated list

            for idx in range(n):
                node_idx_prev = idx
                node_idx_curr = idx + 1
                if node_idx_curr >= n:  # then the ring is closed
                    break
                gdv_i = knearest_global_knn(node_idx_curr, node_idx_prev, r, ldv, gdv, p0, d, k=k)
                gdv.append(gdv_i)
                if DEBUG:
                    print(gdv_i, node_idx_prev, node_idx_curr)

            node_idx_curr = 0
            node_idx_prev = n - 1
            gdv_i = knearest_global_knn(node_idx_curr, node_idx_prev, r, ldv, gdv, p0, d, k=k)
            gdv.append(gdv_i)

        gdv = gdv_i
        gdv = np.array(gdv)
        gdv.sort()
        # At the end of the rounds gdv = real_gdv (CHECK FOR PRIVACY BUGS: maybe with this implementation a node can snoop
        # other nodes' topk)

        real_gdv = []
        for ldv_i in ldv:
            dists = ldv_i[0].flatten()
            real_gdv.append(dists)

        real_gdv = np.array(real_gdv).flatten()
        real_gdv.sort()
        # real_gdv = true_knn.kneighbors(x_test_sample, n_neighbors=k)
        if DEBUG:
            print('REAL TOP K DISTANCES ', real_gdv[:k])
            print('GLOBAL DISTANCE VECTOR ', gdv)

        k_point = gdv[-1]

        lcv = []

        for node_idx in range(n):
            lcv_i = np.zeros((1, len(np.unique(y)))).squeeze().tolist()

            local_knn = local_classifiers[node_idx]
            ldv_i = local_knn.kneighbors(x_test_sample, n_neighbors=k)

            dists = ldv_i[0].flatten()
            ids = ldv_i[1].flatten()

            dist_id = 0
            for dist in dists:
                if dist <= k_point:
                    idx = ids[dist_id]
                    cl = y_split[node_idx][idx]
                    if DEBUG:
                        print(node_idx, idx, cl)
                    lcv_i[cl] += dist

                dist_id += 1

            lcv.append(lcv_i)
            # local_pred = int(local_knn.predict(x_test_sample))
            # lcv_i[local_pred] += 1
            # lcv.append(lcv_i)
            # lcv_i = local_knn.predict_proba(x_test_sample)
            # lcv.append(lcv_i)
            node_idx += 1

        random_values = np.random.randint(100, size=len(np.unique(y)))
        gcv = np.copy(random_values)

        for lcv_i in lcv:
            pos = 0
            for val in lcv_i:
                gcv[pos] += val
                pos += 1

        real_gcv = gcv - random_values

        if DEBUG:
            print('final gcv (what the final node sees) ', gcv)
            print('random values ', random_values)
            print('real gcv ', real_gcv)

        cls = np.argmax(real_gcv)

        if DEBUG:
            print('REAL CLASS: %d\n'
                  'PREDICTED CLASS: %d' % (y_test_dataset[sample_id], cls))

        y_pred.append(cls)
    # plot.print_metrics(y_test_dataset, y_pred)
    return metrics.balanced_accuracy_score(y_test_dataset, y_pred), true_baseline


def test(p0=1.0, d=0.75, rounds=10, k=5):
    # data preparation

    x, y = heart_disease()
    # x = MinMaxScaler().fit_transform(x)
    randomize = np.random.permutation(len(x))
    x = x[randomize]
    y = y[randomize]
    x = np.squeeze(x)
    y = np.squeeze(y)
    n = 3

    x_split = np.array_split(x, n + 1)
    y_split = np.array_split(y, n + 1)

    x_test_dataset = x_split[n]
    y_test_dataset = y_split[n]

    y_pred = []

    for sample_id in range(len(y_test_dataset)):

        if DEBUG:
            print('SAMPLE %d' % (sample_id))

        x_test_sample = x_test_dataset[sample_id].reshape(1, -1)
        ldv = []

        # Each node comptes the distance between x and each point in its database
        for node_idx in range(n):
            dataset = x_split[node_idx]
            ldv_i = []
            tmp = {}

            for point_id in range(len(dataset)):
                point = dataset[point_id]
                d = distance.euclidean(point, x_test_sample)
                tmp[point_id] = d

            sorted_tmp = [(k, tmp[k]) for k in sorted(tmp, key=tmp.get, reverse=False)]
            tmp_dists = []
            tmp_ids = []
            for point_id, dist in sorted_tmp:
                tmp_ids.append(point_id)
                tmp_dists.append(dist)
            tmp_ids = np.array(tmp_ids)
            tmp_dists = np.array(tmp_dists)
            ldv_i.append((tmp_dists[:k], tmp_ids[:k]))
            ldv.append(ldv_i)

        # for node_idx in range(n):
        #     local_knn = local_knn_classifiers[node_idx]
        #     ldv_i = local_knn.kneighbors(x_test_sample, n_neighbors=k)
        #     ldv.append(ldv_i)

        # Now we need to find GLOBALLY the the k nearest distances and store them in gdv

        # gdv = [[np.random.uniform() for _ in range(k + 1)]]

        gdv = [[np.random.uniform(100.0, 200.0) for _ in range(k + 1)]]

        for r in range(rounds):

            # keep only the last item of gdv at next round
            if r > 0:
                gdv = [gdv[-1]]

            if DEBUG:
                print('Round: %d' % r)

            # alg init: curr = prev, then curr=1 and prev = 0
            node_idx_curr = 0
            node_idx_prev = 0
            gdv_i = knearest_global(node_idx_curr, node_idx_prev, r, ldv, gdv, p0, d)
            gdv.append(gdv_i)
            gdv = [gdv[1]]  # remove randomly generated list

            for idx in range(n):
                node_idx_prev = idx
                node_idx_curr = idx + 1
                if node_idx_curr >= n:  # then the ring is closed
                    break
                gdv_i = knearest_global(node_idx_curr, node_idx_prev, r, ldv, gdv, p0, d)
                gdv.append(gdv_i)
                if DEBUG:
                    print(gdv_i, node_idx_prev, node_idx_curr)

            node_idx_curr = 0
            node_idx_prev = n - 1
            gdv_i = knearest_global(node_idx_curr, node_idx_prev, r, ldv, gdv, p0, d)
            gdv.append(gdv_i)

        gdv = gdv_i
        gdv = np.array(gdv)
        gdv.sort()
        # At the end of the rounds gdv = real_gdv (CHECK FOR PRIVACY BUGS: maybe with this implementation a node can snoop
        # other nodes' topk)

        real_gdv = []
        for ldv_i in ldv:
            dists = ldv_i[0][0].flatten()
            real_gdv.append(dists)

        real_gdv = np.array(real_gdv).flatten()
        real_gdv.sort()

        if DEBUG:
            print('REAL TOP K DISTANCES ', real_gdv[:5])
            print('GLOBAL DISTANCE VECTOR ', gdv)

        # sanity check: to which database the top k belong ?
        # if DEBUG:
        #     for v in real_gdv[:5]:
        #         for node_idx in range(n):
        #             data = ldv[node_idx][0].flatten()
        #             if v in data:
        #                 print(v, node_idx)

        # So now every node knows the k nearest distances

        # CLASSIFICATION
        # After each node determines the points in its database which are within the kth nearest distance from x, each
        # node computes a local classification vector of the query instance, where the ith element is the amount of
        # votes the ith class received from the points in this node's database which are among the k  nearest neighbors.
        # Note: so, i need to compute for every node the classification of x ?
        # The nodes then participate to find a global classification vector

        k_point = gdv[-1]

        lcv = []

        for node_idx in range(n):

            lcv_i = np.zeros((1, len(np.unique(y)))).squeeze().tolist()

            # ldv_i = local_knn.kneighbors(x_test_sample, n_neighbors=k)
            ldv_i = ldv[node_idx][0]
            # from the paper isn't very clear. My interpretation:
            # find all points that are in the radius of the k-th points
            dists = ldv_i[0].flatten()
            ids = ldv_i[1].flatten()

            dist_id = 0
            for dist in dists:
                if dist <= k_point:
                    idx = ids[dist_id]
                    cl = y_split[node_idx][idx]
                    if DEBUG:
                        print(node_idx, idx, cl)
                    lcv_i[cl] += 1
                dist_id += 1

            # if DEBUG:
            #     prediction = local_knn.predict(x_test_sample)
            #     my_prediction = np.argmax(lcv_i)
            #     if prediction != my_prediction:
            #         prediction = local_knn.predict(x_test_sample)
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
            pos = 0
            for val in lcv_i:
                gcv[pos] += val
                pos += 1

        real_gcv = gcv - random_values

        if DEBUG:
            print('final gcv (what the final node sees) ', gcv)
            print('random values ', random_values)
            print('real gcv ', real_gcv)

        if real_gcv[0] == real_gcv[1]:  # flip a coin
            cls = 0 if random.random() < 0.5 else 1
        else:
            cls = np.argmax(real_gcv)

        if DEBUG:
            print('REAL CLASS: %d\n'
                  'PREDICTED CLASS: %d' % (y_test_dataset[sample_id], cls))

        y_pred.append(cls)
    # plot.print_metrics(y_test_dataset, y_pred)
    return metrics.balanced_accuracy_score(y_test_dataset, y_pred)


PS = np.arange(0.1, 1.1, 0.1)
DS = np.arange(0.1, 1.1, 0.1)
ROUNDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
DEBUG = True

scores = defaultdict(list)

p = 1.0
d = 0.25

NODES = [3, 4, 5, 6, 7, 8, 9, 10]
NEIGHBORS = [3, 5, 7, 9, 11, 13, 15]
baselines = defaultdict(list)
rounds = {}
score, true_pred = test_knn(p0=p, d=d, rounds=3
                            , k=1, n=3)
print('baseline', true_pred)
print('private knn', score)


def smc_plot():
    for n in ROUNDS:
        print(n)
        for _ in range(100):
            score, true_pred = test_knn(p0=p, d=d, rounds=n, k=5, n=3)
            diff = abs(true_pred - score)  # in the paper they measure ABSOLUTE difference
            scores[n].append(diff)
            baselines[n].append(true_pred)

    pickle.dump(scores, open('smc_scores_d_p100_d25.pkl', 'wb'))
    means = [np.mean(scores[k]) for k in scores]
    dims = [str(r) for r in ROUNDS]
    plt.plot(dims, means)
    plt.xlabel('# rounds')
    plt.ylabel('Average Accuracy Difference')
    plt.title('Private KNN vs Standard KNN\np0 = 10.0, d = 0.25, k = 5')
    plt.savefig('smc_scores_d_p100_d25.png')
    # plt.legend()
    plt.show()


def plot_together():
    scores = pickle.load(open('smc_scores_rounds_p10_d025'))

# smc_plot()
