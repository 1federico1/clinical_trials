import pandas as pd
import os
import numpy as np

cleveland_path = '../data/cleveland.txt'
cleveland_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                     'slope', 'ca', 'thal', 'num']


def heart_disease(is_binary=True):
    my_path = os.path.dirname(__file__)
    path = os.path.join(my_path, cleveland_path)
    cleveland = pd.read_csv(path, sep=',', names=cleveland_columns)

    cleveland_x = cleveland.iloc[:, 0:13]
    cleveland_y = cleveland.num
    cleveland_x_np = cleveland_x.values
    cleveland_y_np = cleveland_y.values
    if is_binary:
        for i in range(len(cleveland_y_np)):
            if cleveland_y_np[i] >= 1:
                cleveland_y_np[i] = 1

    return cleveland_x_np, cleveland_y_np


def skin_noskin():
    path = '/home/federico/Documents/datasets/skin/skin_noskin.txt'
    skin = pd.read_csv(path, sep='\t', names=['B', 'G', 'R', 'label'])

    x = skin.iloc[:, 0:3]
    y = skin.label

    x_np = x.values
    y_np = y.values

    for i in range(len(y_np)):
        if y_np[i] == 1:
            y_np[i] = 0
        elif y_np[i] == 2:
            y_np[i] = 1

    return x_np, y_np


def abalone():
    path = '/home/federico/Documents/datasets/abalone/abalone.txt'
    x = []
    y = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            line = line.split(',')
            row = []
            for attr in line:
                if attr == 'M':
                    attr = 0
                elif attr == 'F':
                    attr = 1
                elif attr == 'I':
                    attr = 2
                row.append(attr)
            x.append(row)
    x = np.array(x, dtype=np.float)
    y = x[:, -1]
    y = np.array(y, dtype=np.int)
    return x, y

