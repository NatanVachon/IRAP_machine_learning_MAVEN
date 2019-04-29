# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:44:17 2019

@author: natan
"""

COST_THRESHOLD = 3*60

import scripts as S

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

def predict2(dataset, model, verbose = True): #TODO enlever qual
    c_dataset = dataset.copy().drop(['epoch', 'x', 'rho', 'deriv_r'], axis = 1)
    if 'label' in dataset.columns:
        c_dataset = c_dataset.drop('label', axis=1)
    np_data = c_dataset.values
#    c_dataset = np.array(dataset.totels_1)
    algo = rpt.Dynp(model = model).fit(np_data)
    breakpoints = algo.predict(n_bkps = 1)
#    breakpoints = algo.predict(epsilon=1e-100) #3e12
#    breakpoints = algo.predict(epsilon=dataset.count()[0] * 1109877913)
    if(verbose):
        rpt.show.display(np_data, breakpoints, figsize=(8, 5))
    return breakpoints

def prediction_comparison_script():
    data = S.create_dataset(shock_nb=1)
    bp_l2 = predict2(data, 'l2', False)
    bp_rbf = predict2(data, 'rbf', False)
    print("l2: " + str(bp_l2[0]))
    print("rbf: " + str(bp_rbf[0]))

    # Choose between the two computed breakpoints
    delta = abs(bp_l2[0] - bp_rbf[0])
    if(delta > COST_THRESHOLD):
        brkp = bp_rbf
    else:
        brkp = bp_l2
    rpt.show.display(data.drop(['epoch', 'r', 'lat', 'long', 'rho', 'deriv_r', 'label'], axis = 1), brkp, figsize=(8, 5))
    return brkp[0]
#Plan: Trouver un breakpoint en passant en rbf et l2, trouver classes, décaler breakpoint si on a pris rbf