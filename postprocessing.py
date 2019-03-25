# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:54:41 2019

This file contains post processing functions
"""

import neural_networks as nn
import pandas as pd

"""
Predict classes according to a ANN
"""
def get_prediction(dataset, ANN, timed_Xtest, timed_ytest):
    timed_ypred = nn.get_pred_timed(ANN, timed_Xtest, dataset.drop(['label'],axis=1))

    raw_proba = nn.get_prob_timed(ANN, timed_Xtest, dataset.drop(['label'],axis=1))

    #variations
    pred_variations = get_var(timed_ypred)
    true_variations = get_var(timed_ytest)

    true_variations = dataset.get_category(true_variations)
    pred_variations = dataset.get_closest_var_by_cat(true_variations, dataset.get_category(pred_variations))

    #crossings reference
    true_crossings = dataset.crossings_from_var(true_variations)

    return timed_ypred, raw_proba, true_variations, pred_variations, true_crossings

"""
Returns a list of variations associated to a y_timed DataFrame (typically y_test or a prediction set)
y_timed : pandas.DataFrame with at least a 'label' and an 'epoch' columns
"""
def get_var(y_timed):
    y_timed = y_timed.sort_values(by='epoch')
    y_timed.index = y_timed.epoch
    var = pd.DataFrame() #cette fois on stocke les variations dans une dataframe avec les classes précédente et suivante
    curr_state = y_timed['label'].iloc[0]
    prec = []
    follow = []
    t = []
    for i in range(y_timed.count()[0] - 1):
        new_state = y_timed['label'].iloc[i+1]
        dt = y_timed['epoch'].iloc[i+1] - y_timed['epoch'].iloc[i]
        if (curr_state != new_state) and dt<60: #si 2 états successifs sont séparés de plus de 1 minute, ce n'est pas une variation mais un trou de données
            t.append(y_timed.index[i])
            prec.append(curr_state)
            follow.append(new_state)
        curr_state = new_state

    var['epoch'] = t
    var['prec_class'] = prec
    var['follow_class'] = follow
    var = var.sort_values(by='epoch')
    var.index = var.epoch
    #for console printing
    print('Total nb. variations: ', var.count()[0])
    return var