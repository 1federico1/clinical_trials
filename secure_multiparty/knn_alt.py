import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from utils.datasets import heart_disease

from random import random

DEBUG = True


def knearest_global_knn(node_idx_curr, node_idx_prev, round, ldv, gdv, p0=1.0, d=0.75, k=5):
    pr = p0 * (d ** (round - 1))
    if DEBUG:
        print('Probability', pr)
    curr_top_k = ldv[node_idx_curr].flatten()

    gdv_prev = gdv[node_idx_prev]

    stack = np.hstack((curr_top_k, gdv_prev))
    stack[::-1].sort()
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
        if random() > pr:
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


def test_knn(p0=1.0, d=0.75, rounds=10, k=5, n=3):
    x, y = heart_disease()
    x = MinMaxScaler().fit_transform(x)
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

        gdv = [[np.random.uniform() for _ in range(k + 1)]]

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
        real_gdv = true_knn.kneighbors(x_test_sample, n_neighbors=k)
        if DEBUG:
            print('REAL TOP K DISTANCES ', real_gdv)
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
    return metrics.balanced_accuracy_score(y_test_dataset, y_pred), true_baseline


x, y = heart_disease()
num_nodes = 3
k = 5  # number of k nearest neighbors
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
global_knn = KNeighborsClassifier(n_neighbors=k)
global_knn.fit(x_train, y_train)
global_y_pred = global_knn.predict(x_test)
baseline = metrics.balanced_accuracy_score(y_test, global_y_pred)
print(baseline)
# Suppose 3 nodes
# For simplicity every nodes has only one sample x
# We have a basic knowledge M
# Every node trains their classifier with M + x. Samples x

ids = np.random.randint(x.shape[0], size=4)

x1, x2, x3, x4 = x[ids]
y1, y2, y3, y4 = y[ids]

x = np.delete(x, ids, axis=0)
y = np.delete(y, ids)

x_samples = [x1, x2, x3]
y_samples = [y1, y2, y3]

x_tests = [x4]
y_tests = [y4]

local_knn = []

for n in range(num_nodes):
    knn = KNeighborsClassifier(n_neighbors=k)
    x_sample = x_samples[n]
    y_sample = y_samples[n]
    # additional knowledge
    x_a = np.vstack([x, x_sample])
    y_a = np.hstack([y, y_sample])
    knn.fit(x_a, y_a)
    local_knn.append(knn)

for test_id in range(len(x_tests)):
    x_test_sample = x_tests[test_id].reshape(-1, 1)
    y_test_sample = y_tests[test_id]

    ldv = []
