import matplotlib.pyplot as plt
import numpy as np


def important_features(importance_ratios):
    plt.bar(np.arange(importance_ratios.shape[0])+1, importance_ratios)
    plt.show()

