# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:54:41 2019

This file contains post processing functions
"""

import neural_networks as nn
import evaluation as ev

import pandas as pd
import statistics as stat
import numpy as np

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
New version of the functions that corrects labels
Idea : instead of correcting variations, we correct labels directly by taking
for each of them a sliding window and assigning the most represented class in this sliding window
"""
def get_corrected_pred(var_ref, y_timed, timed_proba, Dt):
    corr_y = y_timed.copy()
    corr_lab = []
    n = y_timed.count()[0]
    last_account_index = 0

    if var_ref.count().max() < 2 :
        return y_timed

    for i in range(n):
        if(i%10000==0):
            print("Corrected ",i,"/",n)
        curr_t = y_timed.iloc[i]['epoch']
        curr_var_t = var_ref.iloc[last_account_index]['epoch']

        if abs(curr_t - curr_var_t) < Dt - Dt/4 :

            window = y_timed.loc[(y_timed['epoch'] > curr_t - Dt/2) & ((y_timed['epoch'] < curr_t + Dt/2))]

            sw_window = window.loc[window['label']==2]
            sh_window = window.loc[window['label']==1]
            ev_window = window.loc[window['label']==0]

            sw_prob = timed_proba.loc[timed_proba['epoch'].isin(sw_window['epoch'])]['prob_sw']
            sh_prob = timed_proba.loc[timed_proba['epoch'].isin(sh_window['epoch'])]['prob_sh']
            ev_prob = timed_proba.loc[timed_proba['epoch'].isin(ev_window['epoch'])]['prob_ev']

            n_sw = sum(sw_prob)
            n_sh = sum(sh_prob)
            n_ev = sum(ev_prob)

            if n_sh>0:
                if n_sw >= max(n_sh, n_ev):
                    corr_lab.append(2)
                elif n_sh >= max(n_sw, n_ev):
                    corr_lab.append(1)
                else:
                    corr_lab.append(0)
            #si les 2 seules classes sont les classes externes on réintroduit le choc
            #A FAIRE
            else:
                if n_sw > n_ev :
                    if n_sw < n_ev*3 :
                        corr_lab.append(1)
                    else: corr_lab.append(2)
                elif n_sw <= n_ev :
                    if n_ev < n_sw*3 :
                        corr_lab.append(1)
                    else: corr_lab.append(0)
        else:
            corr_lab.append(y_timed.iloc[i]['label'])

        if(curr_t>curr_var_t + Dt - Dt/4):
            last_account_index+=1;
            if last_account_index>var_ref.count()[0] - 2:
                last_account_index = var_ref.count()[0] - 1;

    corr_y['label'] = corr_lab
    return corr_y

"""
Different version of postprocessing on raw probabilities
Instead of comparing the global probability on detected classes, we just compare
the mean probabilities within the sliding window
"""

def get_corrected_pred2(y_timed, timed_proba, Dt):
    corr_y = y_timed.copy()
    delta = int(Dt/4) # Boudary effect offset, 4 corresponds to the 4s timestep
    # Create a window sliding on each point
    for i in range(delta, y_timed.count()[0] - delta):
        mean_probas = np.zeros((3, 1))
        window_probas = timed_proba.iloc[i - delta:i + delta]
        mean_probas[2] = window_probas['prob_sw'].mean()
        mean_probas[1] = window_probas['prob_sh'].mean()
        mean_probas[0] = window_probas['prob_ev'].mean()
        corr_y.at[i, 'label'] = np.argmax(mean_probas)
    return corr_y

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