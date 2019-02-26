import pandas as pd
import numpy as np
import math


def import_data(format=None):
    data = pd.read_csv(
        '/Users/steliosrammos/Documents/Education/Maastricht/DKE-Year3/BachelorThesis/bachelor_thesis/data/interim/data_2018.csv',
        sep=';')
    # data.head()

    X, y, s = None, None, None

    X = data.iloc[:, 1:-2]
    s = data.iloc[:, -2]
    y = data.iloc[:, -1]

    X = X.fillna(X.mean().apply(lambda x: math.floor(x)))
    y.loc[s == 0] = np.nan

    # Nunmpy Arrays
    if format == 'numpy_array':
        X = X.values
        s = s.values
        y = y.values

    return X, s, y