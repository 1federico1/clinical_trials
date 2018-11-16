import random
from collections import defaultdict
import pickle
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from utils import datasets, plot
from utils.grid_search import grid_search, KNN_PARAMS, SVM_PARAMS

EPS_RANGE = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def randomized_response(dataset, eps):
    '''
    :param dataset:  nxd matrix whose rows correspond to users and columns to attributes, in which each value
    is represented in bits
    :param eps: privacy parameter (< 1/2, since flipping bits completely randomly is useless)
    :return: the transformed dataset, arrays p and q for every attribute, in order to do the aggregation (not used for
    classification, since we are going to directly use the perturbed data)
    '''
    # I took the base implementation of the algorithm from here:
    # https://stackoverflow.com/questions/46578629/generalized-random-response-for-local-differential-privacy-implementation?rq=1

    max_values = dataset.max(axis=0)
    min_values = dataset.min(axis=0)

    # p and q for every attribute
    ps = np.empty(dataset.shape)
    qs = np.empty(dataset.shape)

    rand_dataset = []

    value_ranges = []
    for col in range(dataset.shape[1]):
        col_unique_values = np.unique(dataset[:, col])
        value_ranges.append(col_unique_values)

    for x in range(len(dataset)):

        rand_dataset.append([])
        for y in range(len(dataset[x])):
            value_range = value_ranges[y]
            d = max_values[y]

            if len(value_range) == 2:  # binary randomized response
                p = 1 / (1 + np.exp(eps))
                q = 0.
                if random.random() <= p:
                    rand_dataset[x].append(dataset[x][y])
                else:
                    if dataset[x][y] == min_values[y]:
                        rand_dataset[x].append(max_values[y])
                    else:
                        rand_dataset[x].append(min_values[y])
            else:
                p = np.exp(eps) / (np.exp(eps) + d - 1)
                q = 1 / (np.exp(eps) + d - 1)
                if random.random() <= p:
                    rand_dataset[x].append(dataset[x][y])
                else:
                    ans = []
                    if dataset[x][y] == min_values[y]:
                        ans = np.arange(min_values[y] + 1, max_values[y] + 1).tolist()
                    elif dataset[x][y] == max_values[y]:
                        ans = np.arange(min_values[y], max_values[y]).tolist()
                    else:
                        a = np.arange(min_values[y], dataset[x][y]).tolist()
                        b = np.arange(dataset[x][y] + 1, max_values[y]).tolist()
                        [ans.append(i) for i in a]
                        [ans.append(i) for i in b]
                    rand_dataset[x].append(random.choice(ans))

            ps[y] = p
            qs[y] = q

    return np.array(rand_dataset), ps, qs


def aggregate(rand_dataset, col_index, ps, qs, v, n):
    nv = 0  # number of users who answered the value v

    for val in rand_dataset[:, col_index]:  # it is a column
        if val == v:
            nv += 1
    p = ps[col_index][0]
    q = qs[col_index][0]

    iv = nv * p + (n - nv) * q
    estimation = (iv - (n * q)) / (p - q)
    return estimation


def heart_generalized_response(clf=KNeighborsClassifier(), params=KNN_PARAMS, test_size=0.25):
    x, y = datasets.heart_disease()

    x_scale = MinMaxScaler().fit_transform(x)

    gs, x_train, y_train, x_test, y_test = grid_search(x_scale, y, clf, params=params,
                                                       test_size=test_size,
                                                       standardize=False, verbose=1)
    clf = gs.best_estimator_
    y_pred = clf.predict(x_test)
    plot.print_metrics(y_test, y_pred)
    baseline = metrics.balanced_accuracy_score(y_test, y_pred)
    print(baseline)
    num_rounds = 100
    scores = defaultdict(list)

    for eps in EPS_RANGE:
        for i in range(num_rounds):
            z, ps, qs = randomized_response(x, eps)
            z_scale = MinMaxScaler().fit_transform(z)
            z_train, z_test, y_train, y_test = train_test_split(z_scale, y, test_size=test_size)
            clf = gs.best_estimator_
            clf.fit(z_train, y_train)
            y_pred = clf.predict(z_test)
            score = metrics.balanced_accuracy_score(y_test, y_pred)
            scores[eps].append(score)

        print('eps = %.2f, mean = %.10f' % (eps, np.mean(scores[eps])))

    return scores, baseline


if __name__ == '__main__':
    scores, baseline = heart_generalized_response(SVC(), SVM_PARAMS)
    pickle.dump(scores, open('de_svm.pkl', 'wb'))
    pickle.dump(baseline, open('de__svm_baseline.pkl', 'wb'))
    plot.plot_direct_encoding(scores, baseline, filename='de.png', title='DIRECT ENCODING - SVM')