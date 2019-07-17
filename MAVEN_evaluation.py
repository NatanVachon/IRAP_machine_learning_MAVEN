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
from ruptures.metrics import precision_recall

import MAVEN_scripts as S
import preprocessing as prp
import mlp

LIST_PATH = "../Data/datasets/ShockMAVEN_dt1h_list.txt"

"""
Computes metrics comparing predicted shock epochs and a catalog
Inputs:
    epochs: Predicted shock epochs
    dt_tol: Delta time in which we consider the shocks detected
    list_path: Path of the catalog
Returns:
    false_epochs: False positive detected shocks so you can check on AMDA what is wrong
"""
def metrics_from_epoch(epochs, dt_tol, list_path=LIST_PATH):
    epochs = [pd.Timestamp(epochs[i]).timestamp() for i in range(len(epochs))]

    true_data = pd.read_csv(list_path)
    for i in range(true_data.count()[0]):
        true_data.at[i, "epoch"] = pd.Timestamp(true_data.at[i, "epoch"]).timestamp()

    # Get true shocks
    true_shocks = []
    false_epochs = []
    for pred_epoch in epochs:
        index = next((true_data.index[k] for k, true_epoch in enumerate(true_data["epoch"]) if abs(true_epoch - pred_epoch) < 20*60), None)  #Max +/- 30min
        if index is not None:
            true_shocks.append(true_data.at[index, "epoch"])
            true_data = true_data.drop(index)
        else:
            false_epochs.append(pred_epoch)

    # Compute metrics
    acc, recall = 0, 0
    if(len(true_shocks) != 0 and len(epochs) != 0):
        # Add an element so the end of each list is the same for the rupture's function
        if true_shocks[-1] < epochs[-1]:
            true_shocks[-1] = epochs[-1]
        else:
            epochs[-1] = true_shocks[-1]
        # Compute metrics
        acc, recall = precision_recall(true_shocks, epochs, dt_tol)
    else:
        acc, recall = -1, -1
    print("Accuracy: " + str(acc))
    print("Recall: " + str(recall))
    return false_epochs

"""
Same as metrics_from_epoch but with prediction before metrics evaluation
Inputs:
    manager: Training manager containing the model
    data: Data to predict
    dt_corr: Postprocessing time constant
    dt_tol: Detection tolerance
    list_path: Path of the catalog
    postprocessed: Use postprocessing if true
Returns:
    acc: Accuracy
    recall: Recall
"""
def metrics_from_manager(manager, data, dt_corr, dt_tol, list_path=LIST_PATH, postprocessed=True):
    if "label" in data.columns:
        data = data.drop("label", axis=1)
    if postprocessed:
        pred_data = S.corrected_prediction(manager, data, dt_corr, plot=False)
    else:
        pred_data = manager.get_pred(data.drop("epoch", axis=1))
    pred_data["epoch"] = data["epoch"]
    pred_var = get_var(pred_data)
    pred_var = get_category(pred_var)
    pred_cross = crossings_from_var(pred_var)
    # Gather shock epoch
    true_data = pd.read_csv(list_path)
    for i in range(true_data.count()[0]):
        true_data.at[i, "epoch"] = pd.Timestamp(true_data.at[i, "epoch"]).timestamp()

    # Get true shocks
    true_shocks = []
    for pred_epoch in pred_cross["epoch"]:
        index = next((true_data.index[k] for k, true_epoch in enumerate(true_data["epoch"]) if abs(true_epoch - pred_epoch) < 20*60), None)  #Max +/- 30min
        if index is not None:
            true_shocks.append(true_data.at[index, "epoch"])
            true_data = true_data.drop(index)

    # Compute metrics
    acc, recall = 0, 0
    if(len(true_shocks) != 0 and len(pred_cross) != 0):
        # Add an element so the end of each list is the same for the rupture's function
        if true_shocks[-1] < pred_cross.at[pred_cross.count()[0]-1, "epoch"]:
            true_shocks[-1] = pred_cross.at[pred_cross.count()[0]-1, "epoch"]
        else:
            pred_cross.at[pred_cross.count()[0]-1, "epoch"] = true_shocks[-1]
        # Compute metrics
        acc, recall = precision_recall(true_shocks, pred_cross["epoch"], dt_tol)
    else:
        acc, recall = -1, -1
    print("Accuracy: " + str(acc))
    print("Recall: " + str(recall))
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
def crossings_from_var(var):
    begin, end = [], []
    direction = []
    bPass = False
    for i in range(var.count()[0]-1):
        if bPass:
            bPass = False
            continue
        v = var.iloc[i]
        v_next = var.iloc[i+1]
        if v['category'] == 0.5:
            begin.append(v["epoch"])
            end.append(v_next["epoch"])
            if v['follow_class'] == 0:
                direction.append(0)
            else:
                direction.append(1)
        else:
            if v["category"] != -1:
                bPass = True
                dt = v_next['epoch'] - v['epoch']
                if v['follow_class'] == v_next['prec_class'] and v['category']!=v_next['category'] and dt<1200:
                    begin.append(v["epoch"])
                    end.append(v_next["epoch"])
                    if v_next['follow_class'] == 0:
                        direction.append(0)
                    else:
                        direction.append(1)
    crossings = pd.DataFrame()
    crossings["begin"] = begin
    crossings["end"] = end
    crossings['direction'] = direction
    return crossings

"""
Corrects a variations list to reduce the number of quick oscillations by applying the following process:
    Get a variation var_i
    Define a time interval [var_i[t], var_i[t] + Dt]
    For all following variations in this interval, check if they cancel each other
        ex : 2->1->0->2
    Delete all the variations that satisfy this condition

Inputs:
    var : pandas.DataFrame with at least the columns 'epoch', 'prec_class' and 'follow_class'
    Dt  : time interval to consider in seconds
Returns:
    pd.DataFrame Postprocessed variations
"""
def corrected_var(var, Dt):
    epoch_to_skip = []
    i = 0
    while(i<var.count()[0] - 1):
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
Computes crossings using a postprocessing step.
Considering a time constant dt, if two shocks are closer than dt, we convert these two shocks
in a new one which is the average of the two shocks.
This means that the begin epoch of the new shock is the average of the begin epochs of the old shocks
and this is the same for the end epoch

Inputs:
    cross: Raw crossings
    dt: Time constant used to reduce shock nb
Returns:
    final_df: Postprocessed predicted crossings
"""
def corrected_crossings(cross, dt):
    center_epochs = [[0.5 * (cross.at[k, "begin"] + cross.at[k, "end"]), 1] for k in range(len(cross))]
    begins = [cross.at[k, "begin"] for k in range(len(cross))]
    ends = [cross.at[k, "end"] for k in range(len(cross))]

    i = 0
    while i < len(center_epochs):
        indexes_i = [k for k in range(len(center_epochs)) if abs(center_epochs[k][0] - center_epochs[i][0]) <= dt]
        if len(indexes_i) > 1:
            # Compute new values
            new_epoch = [0, 1]
            new_epoch[1] = sum([center_epochs[k][1] for k in indexes_i])
            new_epoch[0] = sum([center_epochs[k][1] * center_epochs[k][0] for k in indexes_i]) / new_epoch[1]
            new_begin = sum([center_epochs[k][1] * begins[k] for k in indexes_i]) / new_epoch[1]
            new_end = sum([center_epochs[k][1] * ends[k] for k in indexes_i]) / new_epoch[1]

            # Remove old values
            epochs_copy = center_epochs.copy()
            begins_copy = begins.copy()
            ends_copy = ends.copy()
            for j in range(len(indexes_i)):
                center_epochs.remove(epochs_copy[j])
                begins.remove(begins_copy[j])
                ends.remove(ends_copy[j])

            # Add new value
            insert_index = min(indexes_i)
            center_epochs.insert(insert_index, new_epoch)
            begins.insert(insert_index, new_begin)
            ends.insert(insert_index, new_end)

        i += 1

    # Save data
    final_df = pd.DataFrame()
    final_df["begin"] = begins
    final_df["end"] = ends
    return final_df

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

"""
Evaluates the different metrics above for a test set and a prediction

This function is really the basics of evaluating the performance of an algorithm based on its predictions,
it is almost independent from the problem itself and uses generic metrics that are widely used in all machine
learning problems.

y_test, y_pred : type pandas.DataFrame, containing at least a 'label' column
"""
def basic_evaluate(y_test, y_pred, verbose=0):
    m = get_confusion_matrices(y_test['label'], y_pred['label'])[0]
    norm_m = get_confusion_matrices(y_test['label'], y_pred['label'])[1]
    p = accuracy_from_cm(m)
    r = recall_from_cm(m)
    f = f_measure_from_cm(m)

    """
    For now, the global p, r and f are valid only for the 3-classes case
    """
    nb0 = y_test.loc[y_test['label']==0].count()[0]
    nb1 = y_test.loc[y_test['label']==1].count()[0]
    nb2 = y_test.loc[y_test['label']==2].count()[0]

    gp = (p[0]*nb0 + p[1]*nb1 + p[2]*nb2)/y_pred.count()[0]
    gr = (r[0]*nb0 + r[1]*nb1 + r[2]*nb2)/y_pred.count()[0]
    gf = (f[0]*nb0 + f[1]*nb1 + f[2]*nb2)/y_pred.count()[0]

    if(verbose==1):
        print('\nClass indices :   0 = Close Environment  or  0 = Close Environment')
        print('                  1 = Bow Shock              1 = inward Bow Shock')
        print('                  2 = Solar Wind             2 = outward Bow Shock')
        print('                                             3 = Solar Wind')

        print('\nPrecisions : ', p)
        print('Recalls    : ', r)
        print('F-measures : ', f)

        print('\nGlobal precision : ', gp)
        print('Global recall    : ', gr)
        print('Global f-measure : ', gf)

        print('\n')
        print('Confusion matrix:\n')
        for i in range(len(norm_m)):
            print(norm_m[i])

    return norm_m, p, r, f

"""
We want to study the influence of parameters that are external to the training on the results of
predictions for the shock. We subdivide this parameter's value range in bins and then compute
the accuracy and recall for each one of those bins

Arguments:
    param_name : str : name of the param to study (included as a column in the shock_data DataFrame?)
    shock_data : DataFrame : basically y_true or y_pred where label==shock
    nb_bins : number of bins to divide the parameters value range into
    show_dist : if True, plots the distribution of the data along the parameter
"""
def split_data_on_param(param_name, data, nb_bins, show_dist=False):
    p_min = data[param_name].min()
    p_max = data[param_name].max()
    bin_width = (p_max - p_min)/nb_bins
    bin_center_vals = [(p_min+(i+0.5)*bin_width) for i in range(nb_bins)]

    sub_data = [[]]*nb_bins
    for i in range(nb_bins):
        curr_shocks = data.loc[data[param_name] > p_min + i*bin_width]
        curr_shocks = curr_shocks.loc[curr_shocks[param_name] < p_min + (i+1)*bin_width]
        sub_data[i] = curr_shocks
    #at this point sub_data is a list of DataFrame corresponding to each bin of the parameter
    #we can then compute evaluation metrics on those subsets

    if show_dist:
        fig, ax = plt.subplots()
        data_dist = [sub_data[i].count()[0] for i in range(len(sub_data))]
#        ax = sns.distplot(data_dist, bins=nb_bins)
        ax.grid(False)
        ax.set_ylabel('Number of data points')
        ax.set_xlabel(param_name)
        ax = plt.plot(bin_center_vals,data_dist)

    ###
    return bin_center_vals,sub_data

"""
Classifies the data in sub datasets (cf previous function) and evaluates each one of them
"""
def perf_on_param(param_name, true_data, pred_data, nb_bins):
    bins,split_tdata = split_data_on_param(param_name, true_data, nb_bins, show_dist=True)
    split_pdata = split_data_on_param(param_name, pred_data, nb_bins)[1]
    acc = []
    rec = []
    f_meas = []
    for i in range(nb_bins):
        if split_tdata[i].loc[split_tdata[i]['label'] == 1.0].count()[0]>0:
            m,p,r,f = basic_evaluate(split_tdata[i], split_pdata[i])
            acc.append(p[1])
            rec.append(r[1])
            f_meas.append(f[1])  #on prend les stats uniquement pour la classe 1 (choc)
        elif i>0:
            acc.append(acc[i-1])
            rec.append(rec[i-1])
            f_meas.append(f_meas[i-1])
        else:
            acc.append(0) #2 as impossible value
            rec.append(0)
            f_meas.append(0)
    fig, ax = plt.subplots()
    ax.set_ylim(0,1)
    ax.set_xlabel(param_name)
    ax.scatter(bins, acc, label = 'Accuracy')
    ax.scatter(bins, rec, label = 'Recall')
    ax.scatter(bins, f_meas, label = 'F-Measure')
    ax.legend()
    return

def k_fold_realtime(manager, k, data, weights=False):
    """
    Runs a k fold validation
    k: Number of folds

    Returns:
        List: One confusion matrix per fold
    """
    cms, cmsPP = [], []
    histories = []
    test_size = 1 / k
    n = data.count()[0]

    for i in range(k):
        print("Fold " + str(i + 1) + '/' + str(k))
        # Run a complete training
        start_i = n * i // k
        timed_data = prp.split_data(data, test_size=test_size, start_index=start_i, ordered=True)
        train_dataset, manager.scaler = prp.scale_and_format(timed_data[0].drop("epoch", axis=1), timed_data[1].drop("epoch", axis=1), timed_data[2], timed_data[3])
        manager.model, history = mlp.run_training(train_dataset, layers_sizes=manager.params["layers_sizes"],layers_activations=manager.params["layers_activations"],
                                                  epochs_nb=manager.params["epochs_nb"], batch_size=manager.params["batch_size"], verbose=1)

        # Compute predictions and metrics
        pred = manager.get_pred(timed_data[1].drop("epoch", axis=1))
        cm = confusion_matrix(timed_data[3]["label"], pred["label"])
        cms.append(cm)

        corr_pred = S.corrected_prediction(manager, timed_data[1], 70, plot=False)
        cm = confusion_matrix(timed_data[3]["label"], corr_pred["label"])
        cmsPP.append(cm)
        histories.append(history)
    return cms, cmsPP, histories

def k_fold_shocks(manager, k, data, weights=False):
    """
    Runs a k fold validation
    k: Number of folds

    Returns:
        List: One confusion matrix per fold
    """
    accs, recalls = [], []
    accsPP, recallsPP = [], []
    test_size = 1 / k
    n = data.count()[0]

    for i in range(k):
        print("Fold " + str(i + 1) + '/' + str(k))
        # Run a complete training
        start_i = n * i // k
        timed_data = prp.split_data(data, test_size=test_size, start_index=start_i, ordered=True)
        train_dataset, manager.scaler = prp.scale_and_format(timed_data[0].drop("epoch", axis=1), timed_data[1].drop("epoch", axis=1), timed_data[2], timed_data[3])
        manager.model, _ = mlp.run_training(train_dataset, layers_sizes=manager.params["layers_sizes"],layers_activations=manager.params["layers_activations"],
                                                  epochs_nb=manager.params["epochs_nb"], batch_size=manager.params["batch_size"], verbose=1)

        acc, recall = metrics_from_manager(manager, timed_data[1], 120, 5*60, postprocessed=False)
        accPP, recallPP = metrics_from_manager(manager, timed_data[1], 120, 5*60, postprocessed=True)
        accs.append(acc)
        recalls.append(recall)
        accsPP.append(accPP)
        recallsPP.append(recallPP)
    return accs, recalls, accsPP, recallsPP