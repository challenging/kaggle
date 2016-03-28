#!/usr/bin/env python

import numpy as np
import matplotlib.pylab as plt

from matplotlib.pyplot import cm 
from matplotlib.font_manager import FontProperties

def barchart(learning_logloss):
    plt.figure(figsize=(13, 7))

    colors = cm.rainbow(np.linspace(0, 1, len(learning_logloss.logloss.keys())))
    for model_name, color in zip(learning_logloss.logloss.keys(), colors):
        values = learning_logloss.logloss[model_name]
        plt.plot([idx for idx in range(0, len(values))], values, color=color, label=model_name)

        for idx in range(0, len(values)):
            plt.text(idx, values[idx], round(values[idx], 4))

    plt.title('Log-loss of the different models in the different fold')
    plt.xlabel('No. of Fold')
    plt.ylabel('Log-loss')
    plt.grid(True)

    fontP = FontProperties()
    fontP.set_size('small')
    plt.legend(loc=8, ncol=3, fancybox=True, shadow=True, prop=fontP)

    plt.ylim((0.45, 0.65))
    plt.show()

if __name__ == "__main__":
    import pickle

    learning_logloss = None
    with open("logloss.pickle") as INPUT:
        learning_logloss = pickle.load(INPUT)

    barchart(learning_logloss)
