# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:44:17 2019

@author: natan
"""

import ruptures as rpt
import matplotlib.pyplot as plt
import numpy as np

def predict(dataset):
    c_dataset = dataset.copy()
    c_dataset = c_dataset.drop('label', axis = 1).drop('epoch', axis = 1)
    model = 'l2'
#    algo = rpt.BottomUp(model = model, min_size=75).fit(c_dataset)
    algo = rpt.Dynp(model = model).fit(c_dataset)
    #algo = rpt.Window(width = 15, model = model).fit(c_dataset)
    breakpoints = algo.predict(n_bkps = 2)
#    breakpoints = algo.predict(epsilon=7e11)

    plt.plot(dataset.index, dataset.totels_1, 'b-')
    for i in range(len(breakpoints)):
        plt.plot([breakpoints[i]] * 2, [0, max(dataset.totels_1)], 'k--')

    return breakpoints

def predict2(dataset):
    c_dataset = dataset.copy().drop(['epoch', 'r', 'lat', 'long', 'rho', 'deriv_r'], axis = 1)
#    c_dataset = np.array(dataset.totels_1)
    model = 'l2'
#    algo = rpt.BottomUp(model = model, min_size=75).fit(c_dataset)
#    algo = rpt.Dynp(model = model, min_size=120).fit(c_dataset)
#    algo = rpt.Window(width = 30, model = model).fit(c_dataset)
    algo = rpt.Binseg(model = model, min_size=120).fit(c_dataset)
    breakpoints = algo.predict(n_bkps = 6)
#    breakpoints = algo.predict(epsilon=1e-100) #3e12
#    breakpoints = algo.predict(epsilon=dataset.count()[0] * 1109877913)

    rpt.show.display(c_dataset, breakpoints, figsize=(10, 6))
    return breakpoints