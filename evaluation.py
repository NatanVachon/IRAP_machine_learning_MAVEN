# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:13:30 2019
This file contains everything about data evaluation

@author: natan
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np

LIST_PATH = '../Data/datasets/ShockMAVEN_dt1h_list.txt'

"""
"""
def metrics_with_tolerances(true_data, pred_data, dt_tol):
    true_var, pred_var = get_var(true_data), get_var(pred_data)
    true_var, pred_var = get_category(true_var), get_category(pred_var)
    true_cross, pred_cross = crossings_from_var(true_var), crossings_from_var(pred_var)
    # Apply tolerance on epochs
    acc, recall = acc_recall_from_epochs(true_cross, pred_cross, dt_tol)
    print("Accuracy: " + str(acc))
    print("Recall: " + str(recall))
    return acc, recall

"""
"""
def metrics_from_list(pred_data, dt_tol, list_path=LIST_PATH, lerp_delta=0.5):
    pred_var = get_var(pred_data)
    pred_var = get_category(pred_var)
    pred_cross = crossings_from_var(pred_var, lerp_delta)
    # Gather shock epoch
    true_data = pd.read_csv(list_path)
    for i in range(true_data.count()[0]):
        true_data.at[i, "epoch"] = pd.Timestamp(true_data.at[i, "epoch"]).timestamp()
    # Find begin and end index
    begin_epoch = next(i for i, e in true_data.iterrows() if e["epoch"] > pred_data.at[0, "epoch"])
    end_epoch = next(i - 1 for i, e in true_data.iterrows() if e["epoch"] > pred_data.at[pred_data.count()[0]-1, "epoch"])
    true_cross = true_data.loc[begin_epoch:end_epoch,:]
    true_cross.index = [i for i in range(true_cross.count()[0])]
    acc, recall = acc_recall_from_epochs(true_cross, pred_cross, dt_tol)
    print("Accuracy: " + str(acc))
    print("Recall: " + str(recall))
    return acc, recall

"""
"""
def acc_recall_from_epochs(true_data, pred_data, dt_tol):
    # Apply tolerance on epochs
    true_pos, false_pos, false_neg = 0, 0, 0
    c_true_data = true_data.copy()
    for i, pred_epoch in enumerate(pred_data["epoch"]):
        index = next((j for j, true_epoch in enumerate(c_true_data["epoch"]) if abs(pred_epoch - true_epoch) < dt_tol), None)
        if(index is not None):
            true_pos += 1
            c_true_data = c_true_data.drop(c_true_data.index[index])
        else:
            false_pos += 1
    false_neg = true_data.count()[0] - true_pos
    acc = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    return acc, recall

"""
Plots the graph of classes transitions from a true variations list and a predicted variations list.

true_var, pred_var : pandas.DataFrame with columns 'epoch', 'pred_class', 'follow_class'
data_name : name of the prediction dataset for the plot title
"""
def graph_pred_from_var(true_var, pred_var, data_name=''):
    fig, ax = plt.subplots()
    ax.set_ylim(-0.5,2.5)
    ax.set_xlabel('Time (epoch in s)')
    ax.set_ylabel('Class')

    true_toplot = [[],[]]
    #first list for the epoch, second for the classes
    for i in range(true_var.count()[0]):
        epoch = true_var['epoch'].iloc[i]
        prec_class, follow_class = true_var['prec_class'].iloc[i], true_var['follow_class'].iloc[i]
        true_toplot[0].append(epoch)
        true_toplot[0].append(epoch)
        true_toplot[1].append(prec_class)
        true_toplot[1].append(follow_class)

    pred_toplot = [[],[]]
    #first list for the epoch, second for the classes, third for dt_to_closest var
    for i in range(pred_var.count()[0]):
        epoch = pred_var['epoch'].iloc[i]
        prec_class, follow_class = pred_var['prec_class'].iloc[i], pred_var['follow_class'].iloc[i]
        pred_toplot[0].append(epoch)
        pred_toplot[0].append(epoch)
        pred_toplot[1].append(prec_class)
        pred_toplot[1].append(follow_class)


    ax.plot(pd.to_datetime(true_toplot[0], unit='s'), true_toplot[1], linestyle='--', linewidth = 2, color = 'green')
    ax.plot(pd.to_datetime(pred_toplot[0], unit='s'), pred_toplot[1], color = 'red', linewidth=0.9)
    ax.set_ylim(-0.5,2.5)
    fig.suptitle(data_name + '\nClass representation')
    plt.show()
    return

"""
Returns the confusion matrix and the normalized confusion matrix from a prediction compared to a test set

y_test, y_pred : type list
"""
def get_confusion_matrices(y_test, y_pred):
    m = confusion_matrix(y_test, y_pred)
    normalized_m = m.astype('float') / m.sum(axis=1)[:, np.newaxis]
    return m, normalized_m

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
    #var.index = var.epoch JFD
    #for console printing
    print('Total nb. variations: ', var.count()[0])
    return var


"""
Same function as get_category but for any number of classes
Number of combinations for n variations = sum(k=[1,n])(k) = n*(n+1)/2

var_cat = -1 for non physical variations
"""
def get_category(var, nb_class = 3):
    cat = []
    n = var.count()[0]
    for i in range(n):
        classes = [var['prec_class'].iloc[i], var['follow_class'].iloc[i]]
        classes.sort()
        if classes[0]+1!=classes[1]:
            var_cat = -1
        else:
            var_cat = nb_class*classes[0] + classes[1]
        cat.append(var_cat)
    new_var = var.copy()
    new_var['category'] = cat
    return new_var

"""
Takes a list of variations and returns a list of shock crossings associated
Process:
    if non physical var : just add the epoch of the variation
    if shock detected : takes the middle of the detected shock
'category' needed
Returns a dataframe of crossings with their epoch and direction : 0 for inbound, 1 for outbound
"""
def crossings_from_var(var, lerp_delta=0.5):
    epochs = []
    direction = []
    bPass = False
    for i in range(var.count()[0]-1):
        if bPass:
            bPass = False
            continue
        v = var.iloc[i]
        v_next = var.iloc[i+1]
        if v['category'] == 0.5:
            epochs.append(v['epoch'])
            if v['follow_class'] == 0:
                direction.append(0)
            else:
                direction.append(1)
        else:
            if v["category"] != -1:
                bPass = True
                dt = v_next['epoch'] - v['epoch']
                if v['follow_class'] == v_next['prec_class'] and v['category']!=v_next['category'] and dt<1200:
                    t = v['epoch'] + dt * lerp_delta
                    epochs.append(t)
                    if v_next['follow_class'] == 0:
                        direction.append(0)
                    else:
                        direction.append(1)
    crossings = pd.DataFrame()
    crossings['epoch'] = epochs
    crossings['direction'] = direction
    return crossings

"""
Returns the accuracy, recall and f-measure based on the confusion matrix

matrix : type matrix (defined by the return of get_confusion_matrices)
"""
#Pour se rappeler rapidement de la signification de la précision et du recall:
#Precision:
#    Nombre d'éléments correctement identifiés parmi ceux existants (quelle proportion de prédictions c correspondent a un élément réel c)
#Recall:
#    Nombre d'éléments corrects parmi ceux identifiés (quelle proportion d'éléments réels c sont classifiés c)

def accuracy_from_cm(matrix):
    acc = []
    for i in range(matrix.shape[0]):
        p = matrix[i][i]
        den = 0
        for j in range(matrix.shape[0]):
            den+=matrix[j][i]
        if(den>0):
            acc.append(p/den)
        else:
            acc.append(0)
    return acc

def recall_from_cm(matrix):
    recall = []
    for i in range(matrix.shape[0]):
        p = matrix[i][i]
        den = 0
        for j in range(matrix.shape[1]):
            den+=matrix[i][j]
        if(den>0):
            recall.append(p/den)
        else:
            recall.append(0)
    return recall

def f_measure_from_cm(matrix):
    p = accuracy_from_cm(matrix)
    r = recall_from_cm(matrix)
    f_mes = []
    for i in range(len(p)):
        f_mes.append(2*p[i]*r[i]/(p[i] + r[i]))
    return f_mes