from collections import defaultdict
import pickle
import random
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from utils import datasets, random_projection, plot
from utils.grid_search import grid_search, SVM_PARAMS, KNN_PARAMS

EPS_RANGE = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def jl_dp_accuracy(clf=KNeighborsClassifier(), params = KNN_PARAMS, test_size=0.25, is_sparse=True):
    x, y = datasets.heart_disease()

    x = MinMaxScaler().fit_transform(x)
    split_random_state = int(random.random() * 100)
    gs, x_train, y_train, x_test, y_test = grid_search(x, y, clf, params,
                                                             test_size=test_size,
                                                             standardize=False, verbose=1,
                                                             random_state=split_random_state)
    clf = gs.best_estimator_
    y_pred = clf.predict(x_test)
    plot.print_metrics(y_test, y_pred)
    baseline = metrics.balanced_accuracy_score(y_test, y_pred)
    print(baseline)
    num_rounds = 100
    dims = x.shape[1]
    scores = defaultdict(list)
    delta = 0.1

    for eps in EPS_RANGE:
        print('epsilon = ', eps)
        for k in range(1, dims + 1):
            tmp_k = []
            for i in range(num_rounds):
                z, p, sigma = random_projection.private_projection(x, eps=eps, delta=delta, k=k, is_sparse=is_sparse)
                z_train, z_test, y_train, y_test = train_test_split(z, y, test_size=test_size)
                clf = gs.best_estimator_
                clf.fit(z_train, y_train)
                y_pred = clf.predict(z_test)
                score = metrics.balanced_accuracy_score(y_test, y_pred)
                tmp_k.append(score)

            mean_k = np.mean(tmp_k)
            print('k = %d, mean = %.10f' % (k, mean_k))
            scores[eps].append(mean_k)

    return scores, baseline

if __name__ == '__main__':
    scores, baseline = jl_dp_accuracy(clf=KNeighborsClassifier(), params=KNN_PARAMS, test_size=0.25, is_sparse=False)
    plot.plot_all_epsilons(scores, baseline, EPS_RANGE, filename='jl_dp_gaussian.png')