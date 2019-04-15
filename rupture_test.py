# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:44:17 2019

@author: natan
"""

import ruptures as rpt
import numpy as np

def predict(dataset):

    # Only keep relevant data
    c_dataset = dataset.copy().drop(['epoch', 'r', 'lat', 'long', 'rho', 'deriv_r'], axis = 1)
    if 'label' in dataset.columns:
        c_dataset = c_dataset.drop('label', axis=1)
    n = c_dataset.count()[0]
    model = 'l2'
    algo = rpt.Binseg(model = model).fit(c_dataset)
    breakpoints = algo.predict(n_bkps = 2)
    #breakpoints = algo.predict(epsilon=3e8*n)

    rpt.show.display(c_dataset, breakpoints, figsize=(10, 6))
    print("Shock length : " + str((breakpoints[1] - breakpoints[0])*4/60))
    return breakpoints

def predict2(dataset, model):
    c_dataset = dataset.copy().drop(['epoch', 'r', 'lat', 'long', 'rho', 'deriv_r'], axis = 1)
    if 'label' in dataset.columns:
        c_dataset = c_dataset.drop('label', axis=1)
    np_data = c_dataset.values
#    c_dataset = np.array(dataset.totels_1)
    algo = rpt.Dynp(model = model).fit(np_data)
    breakpoints = algo.predict(n_bkps = 1)
#    breakpoints = algo.predict(epsilon=1e-100) #3e12
#    breakpoints = algo.predict(epsilon=dataset.count()[0] * 1109877913)

    rpt.show.display(np_data, breakpoints, figsize=(10, 6))
    return breakpoints