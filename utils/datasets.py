import pandas as pd
import os

cleveland_path = '../data/cleveland.txt'
cleveland_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak',
                     'slope', 'ca', 'thal', 'num']


def heart_disease(is_binary=True):
    my_path = os.path.dirname(__file__)
    print(my_path)
    path = os.path.join(my_path, cleveland_path)
    print(path)
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
