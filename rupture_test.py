# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:44:17 2019

@author: natan
"""

import ruptures as rpt
import matplotlib.pyplot as plt

def predict(dataset):
    c_dataset = dataset.copy()
    c_dataset = c_dataset.drop('label', axis = 1).drop('epoch', axis = 1)
    model = 'l2'
##    algo = rpt.BottomUp(model = model, min_size=75).fit(c_dataset)
    algo = rpt.Dynp(model = model).fit(c_dataset)
    #algo = rpt.Window(width = 15, model = model).fit(c_dataset)
    breakpoints = algo.predict(n_bkps = 2)
#    breakpoints = algo.predict(epsilon=7e11)

    plt.plot(dataset.index, dataset.totels_1, 'b-')
    for i in range(len(breakpoints)):
        plt.plot([breakpoints[i]] * 2, [0, max(dataset.totels_1)], 'k--')

    return breakpoints