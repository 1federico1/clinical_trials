import itertools
import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def print_metrics(y_true, y_pred):
    cm = metrics.confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, classes=np.unique(y_true))
    print('Balanced Accuracy = %.2f\n'
          'Non-Balanced Accuracy = %.2f' % (
              metrics.balanced_accuracy_score(y_true, y_pred), metrics.accuracy_score(y_true, y_pred)))


def plot_all_epsilons(scores, baseline, eps_range, dims=None, figsize=(15, 15), filename='tmp.png'):
    if dims is None:
        dims = list(range(1, 14))

    baselines = [baseline for _ in dims]
    index = 1
    f = plt.figure(figsize=figsize)
    plt.rc('text', usetex=True)

    for eps in eps_range:
        means = np.mean(scores[eps])
        ax = f.add_subplot(4, 2, index)
        ax.plot(dims, means)
        ax.plot(dims, baselines)
        ax.xaxis.set_ticks(np.arange(1, 13, 1.0))
        ax.set_xlabel('Dimensions')
        ax.set_ylabel('Accuracy')
        ax.set_title('$\epsilon$ = %s ' % (str(eps)))
        index += 1
    plt.savefig(filename)
    plt.show()

def plot_direct_encoding(scores, baseline, filename='tmp.png', title='test'):
    baselines = [baseline for _ in scores]
    means = [np.mean(scores[eps]) for eps in scores]
    vars = [np.var(scores[eps]) for eps in scores]
    dims = [str(eps) for eps in scores]
    plt.rc('text', usetex=True)
    plt.plot(dims, means, label='Mean accuracy')
    plt.errorbar(dims, means, vars, linestyle='None', marker='^', label='Variance of accuracy')
    plt.plot(dims, baselines, label='Baseline')
    plt.legend()
    plt.xlabel('Privacy budget $\epsilon$')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.savefig(filename)
    plt.show()