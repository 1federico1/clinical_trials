import math
import operator
import numpy as np
from random import random
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn import metrics
from utils.datasets import heart_disease

DEBUG = True

x, y = load_breast_cancer(return_X_y=True)
y = y.reshape(-1, 1)
x = np.hstack((x, y))


def train_test_split(dataset, split):
    x_train = []
    x_test = []
    y_test = []
    for x in range(len(dataset) - 1):
        for y in range(30):
            dataset[x][y] = float(dataset[x][y])
        if random() < split:
            x_train.append(dataset[x])
        else:
            x_test.append(dataset[x])
    return x_train, x_test


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def neighbors(train, test_instance, k=5):
    distances = []
    length = len(test_instance) - 1
    for x in range(len(train)):
        dist = euclideanDistance(test_instance, train[x], length)
        distances.append((train[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = [distances[x][0] for x in range(k)]
    return neighbors


def response(neighbors, k_point):
    classvotes = {}
    for x in range(len(neighbors)):
        if np.mean(neighbors[x]) <= k_point:
            response = neighbors[x][-1]
            if response in classvotes:
                classvotes[response] += 1
            else:
                classvotes[response] = 1
    sorted_votes = sorted(classvotes.items(), key=operator.itemgetter(1), reverse=False)
    return sorted_votes[0][0]


y_pred = []
x_train, x_test = train_test_split(x, split=0.67)
for x in range(len(x_test)):
    neighs = neighbors(x_train, x_test[x])
    res = response(neighs, k_point=np.inf)
    y_pred.append(res)

x_test = np.array(x_test)
y_test = x_test[:, 30]

print('Non balanced accuracy', metrics.accuracy_score(y_test, y_pred))
print('Balanced accuracy', metrics.balanced_accuracy_score(y_test, y_pred))


def randomization(p0, d, r):
    return p0 * (d ** (r - 1))


def knearest_global_knn(node_idx_curr, node_idx_prev, round, ldv, gdv, p0=1.0, d=0.75, k = 5):
    pr = randomization(p0, d, round)
    if DEBUG:
        print('Probability', pr)
    curr_top_k = ldv[node_idx_curr].flatten()

    gdv_prev = gdv[node_idx_prev]

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
    x, y = load_breast_cancer(return_X_y=True)
    y = y.reshape(-1, 1)
    x = np.hstack((x, y))
    randomize = np.random.permutation(len(x))
    x = x[randomize]
    x = np.squeeze(x)

    x_split = np.array_split(x, n + 1)

    x_test_dataset = x_split[n]

    y_pred = []

    for sample_id in range(len(x_test_dataset)):

        if DEBUG:
            print('SAMPLE %d' % (sample_id))

        x_test_sample = x_test_dataset[sample_id].reshape(1, -1)
        ldv = []

        for node_idx in range(n):
            node_data = x_split[node_idx]
            ldv_i = np.array(neighbors(node_data, x_test_sample, k))
            ldv.append(ldv_i)

        gdv = [[np.random.uniform(0.1, 100.0) for _ in range(k + 1)]]

        for r in range(rounds):

            # keep only the last item of gdv at next round
            if r > 0:
                gdv = [gdv[-1]]

            if DEBUG:
                print('Round: %d' % r)

            # alg init: curr = prev, then curr=1 and prev = 0
            node_idx_curr = 0
            node_idx_prev = 0
            gdv_i = knearest_global_knn(node_idx_curr, node_idx_prev, r, ldv, gdv, p0, d)
            gdv.append(gdv_i)
            gdv = [gdv[1]]  # remove randomly generated list

            for idx in range(n):
                node_idx_prev = idx
                node_idx_curr = idx + 1
                if node_idx_curr >= n:  # then the ring is closed
                    break
                gdv_i = knearest_global_knn(node_idx_curr, node_idx_prev, r, ldv, gdv, p0, d)
                gdv.append(gdv_i)
                if DEBUG:
                    print(gdv_i, node_idx_prev, node_idx_curr)

            node_idx_curr = 0
            node_idx_prev = n - 1
            gdv_i = knearest_global_knn(node_idx_curr, node_idx_prev, r, ldv, gdv, p0, d)
            gdv.append(gdv_i)

        gdv = gdv_i
        gdv = np.array(gdv)
        gdv.sort()

        k_point = gdv[-1]

        lcv = []

        for node_idx in range(n):

            lcv_i = np.zeros((1, len(np.unique(y)))).squeeze().tolist()

            ldv_i = ldv[node_idx]

            res = int(response(ldv_i, k_point=k_point))
            lcv_i[res] += 1
            lcv.append(lcv_i)
            node_idx += 1

        random_values = np.random.randint(100, size=len(np.unique(y)))

        gcv = np.copy(random_values)

        for lcv_i in lcv:
            pos = 0
            for val in lcv_i:
                gcv[pos] += val
                pos += 1

        if DEBUG:
            real_gcv = gcv - random_values
        else:
            real_gcv = gcv - random_values

        if real_gcv[0] == real_gcv[1]:  # flip a coin
            cls = 0 if random.random() <= 0.5 else 1
        else:
            cls = np.argmax(real_gcv)

        y_pred.append(cls)
    # plot.print_metrics(y_test_dataset, y_pred)
    return y_pred, np.array(x_test_dataset[:, 30])

y_pred, y_test = test_knn(p0=1.0, d=0.5, rounds=100)

print('Non balanced accuracy', metrics.accuracy_score(y_test, y_pred))
print('Balanced accuracy', metrics.balanced_accuracy_score(y_test, y_pred))