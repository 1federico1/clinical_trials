from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

SVM_PARAMS = {'kernel': ['linear', 'rbf', 'sigmoid', 'poly'], 'shrinking': [True, False], 'probability': [True, False],
              'C': [0.01, 0.1, 1.0, 10.0], 'gamma': ['scale', 0.1, 0.01]}

KNN_PARAMS = {'n_neighbors': [7, 9, 11, 15, 25, 49, 75, 99], 'weights': ['uniform', 'distance'],
              'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'leaf_size': [30, 60, 90, 15], 'p': [1, 2]}


def grid_search(x, y, clf, params, test_size=0.33, standardize=False, cv=10, verbose=2, random_state=42):
    """

    :param x:
    :param y:
    :param clf:
    :param params:
    :param test_size:
    :param standardize:
    :return: grid_search object, x_train, y_train, x_test, y_test
    """
    if standardize:
        ss = StandardScaler()
        x = ss.fit_transform(x)

    gs = GridSearchCV(clf, param_grid=params, scoring=metrics.make_scorer(metrics.matthews_corrcoef), n_jobs=-1, cv=cv,
                      verbose=verbose)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
    gs.fit(x_train, y_train)
    print(gs.best_params_)
    return gs, x_train, y_train, x_test, y_test
