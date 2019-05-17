# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:54:41 2019

This file contains post processing functions
"""

import MAVEN_neural_networks as nn
import MAVEN_evaluation as ev

import pandas as pd
import statistics as stat
import numpy as np
import collections
import time

"""
Based on a density of crossings, returns a list of dates associated to the peaks and
a list of "degrees" of crossings (number of multiple crossings around the said date)
"""
def final_list(y_density):
    dates = []
    degrees = []
    to_consider = y_density.loc[y_density['density']>0]
    i=0
    n = to_consider.count()[0]
    while i<n-1:
        start_t = to_consider['epoch'].iloc[i]
        curr_t = to_consider['epoch'].iloc[i+1]
        j = 1
        interval_dates = []
        interval_degree = 1
        while(curr_t - start_t < 10) & (i+j<n):
            deg = to_consider['density'].iloc[i+j]
            interval_dates.extend([curr_t]*deg)
            start_t = curr_t
            curr_t = to_consider['epoch'].iloc[i+j]
            if deg>interval_degree:
                interval_degree = deg
            j+=1
        if len(interval_dates)>0:
#            dates.append(sum(interval_dates)/len(interval_dates))
            dates.append(stat.median(interval_dates))
        else:
            dates.append(curr_t)
        degrees.append(interval_degree)
        i+=j
#        print(i)
    final = pd.DataFrame()
    final['epoch'] = dates
    final['degree'] = degrees
    return final

"""
Different version of postprocessing on raw probabilities
Instead of comparing the global probability on detected classes, we just compare
the mean probabilities within the sliding window
"""
def get_corrected_pred(timed_proba, Dt):
    n = timed_proba.count()[0]
    corr = pd.DataFrame()
    corr["epoch"] = timed_proba["epoch"]
    corr["label"] = [0 for i in range(n)]
    # Boudary effect offset, 4 corresponds to the 4s timestep
    delta = int(Dt/4/2)
    i_delta = 1.0 / (2.0 * delta + 1)
    # Initialize queues
    mean_probas = np.zeros((3, 1))
    first_rows = timed_proba.iloc[0:2 * delta]
    sw_queue = collections.deque(first_rows["prob_sw"], maxlen=2 * delta)
    sh_queue = collections.deque(first_rows["prob_sh"], maxlen=2 * delta)
    ev_queue = collections.deque(first_rows["prob_ev"], maxlen=2 * delta)

    sw_list, sw_i = [e * i_delta for i, e in first_rows["prob_sw"]], 0
    sh_list, sh_i = [e * i_delta for i, e in first_rows["prob_sh"]], 0
    ev_list, ev_i = [e * i_delta for i, e in first_rows["prob_ev"]], 0
    mean_probas[2] = np.sum(sw_list)
    mean_probas[1] = np.sum(sh_list)
    mean_probas[0] = np.sum(ev_list)
    k = len(ev_list)

    begin = time.time()
    for i in range(delta, n - delta):
        # Compute mean probas
#        mean_probas[2] = np.mean(sw_queue)
#        mean_probas[1] = np.mean(sh_queue)
#        mean_probas[0] = np.mean(ev_queue)
        # Get argmax
        corr.at[i, 'label'] = np.argmax(mean_probas)
        # Update queues
#        sw_queue.append(timed_proba.at[i + delta, "prob_sw"])
#        sh_queue.append(timed_proba.at[i + delta, "prob_sh"])
#        ev_queue.append(timed_proba.at[i + delta, "prob_ev"])
        mean_probas[2] = mean_probas[2] - sw_list[sw_i] + timed_proba.at[i + delta, "prob_sw"]
        mean_probas[1] = mean_probas[1] - sh_list[sh_i] + timed_proba.at[i + delta, "prob_sh"]
        mean_probas[0] = mean_probas[0] - ev_list[ev_i] + timed_proba.at[i + delta, "prob_ev"]

        sw_list[sw_i] = timed_proba.at[i + delta, "prob_sw"]
        sw_i = (sw_i + 1) % k
        sw_list[sh_i] = timed_proba.at[i + delta, "prob_sh"]
        sw_i = (sh_i + 1) % k
        sw_list[ev_i] = timed_proba.at[i + delta, "prob_ev"]
        sw_i = (ev_i + 1) % k
    print(time.time() - begin)
    # Fill boundary labels
    corr.loc[:delta, "label"] = corr.at[delta, "label"]
    corr.loc[n - delta:, "label"] = corr.at[n - delta - 1, "label"]
    return corr

"""
Based on a list of crossings and a Dt (sliding window),
we define a crossings density based on the number of crossings in each window
Returns a list of crossings with a new columns

Résolution temporelle des chocs
"""
def crossings_density(y_timed, cross, Dt):
    new_y = y_timed.copy()
    new_y['density'] = 0
    for i in range(cross.count()[0]):
        curr_t = cross.iloc[i]['epoch']
        new_y.loc[(new_y['epoch'] > curr_t - Dt/2) & (new_y['epoch'] < curr_t + Dt/2),'density'] += 1
    return new_y

"""
"""
def corrected_var(var, Dt):
    epoch_to_skip = []
    i = 0
    while(i<var.count()[0] - 1):
#    for i in range(var.count()[0]-1):
        t = var['epoch'].iloc[i]
        t_it = var['epoch'].iloc[i+1]
        start_class = var['prec_class'].iloc[i]

        furthest = 0 #indice de la variation la plus lointaine a supprimer (a partir de i)
        j = 0 #nb de tours de boucle effectués
        while(t_it<t+Dt and i+j<var.count()[0]-2):
            curr_class = var['follow_class'].iloc[i+1+j]
            if curr_class == start_class:
                furthest = j+1
            j += 1
            t_it = var['epoch'].iloc[i+1+j]
        to_skip = []
        if furthest>0:
            to_skip = var['epoch'].iloc[i:i+furthest+1]

        epoch_to_skip.extend(to_skip)
        i = i+1+furthest
    clean_var = var.loc[~var['epoch'].isin(epoch_to_skip)]
    return clean_var

"""
Predict classes according to a ANN
"""
def get_prediction(dataset, ANN, timed_Xtest, timed_ytest):
    timed_ypred = nn.get_pred_timed(ANN, timed_Xtest, dataset.drop(['label'],axis=1))

    raw_proba = nn.get_prob_timed(ANN, timed_Xtest, dataset.drop(['label'],axis=1))

    #variations
    pred_variations = get_var(timed_ypred)
    true_variations = get_var(timed_ytest)

    true_variations = ev.get_category(true_variations)
    pred_variations = ev.get_closest_var_by_cat(true_variations, ev.get_category(pred_variations))

    #crossings reference
    true_crossings = ev.crossings_from_var(true_variations)

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