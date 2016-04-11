#!/usr/bin/env python

import sys

import numpy as np
import matplotlib.pylab as plt

from matplotlib.pyplot import cm
from matplotlib.font_manager import FontProperties

from load import load_cache

def barchart(learning_cost):
    plt.figure(figsize=(13, 7))

    colors = cm.rainbow(np.linspace(0, 1, len(learning_cost.cost.keys())))
    for model_name, color in zip(learning_cost.cost.keys(), colors):
        values = learning_cost.cost[model_name]
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

    plt.ylim((0.45, 0.52))
    plt.show()

if __name__ == "__main__":
    learning_cost = load_cache(sys.argv[1])

    barchart(learning_cost)
