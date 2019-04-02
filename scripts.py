# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:18:13 2019

This file containts scripts
@author: natan
"""

import pandas as pd
import random as rd
import matplotlib.pyplot as plt

import neural_networks as nn
import preprocessing as prp
import postprocessing as pop
import evaluation as ev
import communication_AMDA as acom
import prediction as pred

BEGIN_DATE = '2016-02-16T02:00:31'
END_DATE = '2016-02-16T04:00:31'
PARAMETER_NAMES = ["mav_mars_r", "mav_lat_iaumars", "mav_lon_iaumars", "ws_0", "ws_1", "ws_4", "ws_2", "ws_3", "mav_swiakp_vmso(0)"]
PARAMETER_COLS = [["epoch", "r"], ["epoch", "lat"], ["epoch", "long"], ["epoch", "rho"], ["epoch", "deriv_r"], ["epoch", "mag_var"], ["epoch", "totels_1"], ["epoch", "totels_8"], ["epoch", "SWIA_vel_x"]]
SAVE_PATH = "d:/natan/Documents/IRAP/Data/datasets/"
SHOCK_LIST_PATH = '../Data/datasets/ShockMAVEN_list.txt'

DATASET_PATH = '../Data/datasets/simple_dataset.txt'

MODEL_PATH = '../Data/models/'
FEATURE_NB = 9
CLASS_NB = 3
EPOCHS_NB = 50
BATCH_SIZE = 256
TEST_SIZE = 0.3

LAYERS_SIZES = [FEATURE_NB, 66, 22, CLASS_NB]
LAYERS_ACTIVATIONS = ['relu', 'relu', 'tanh', 'softmax']

"""
Running a k-fold validation with the defined parameters
Evaluating the network for the following metrics:
    - initial network f-measure
    - shocks identification f-measure at Dt = 300s
    - shocks identification f-measure at Dt = 600s
    - ratio nb. predicted variations / true variations
    - loss (jaccard)
Returns a DataFrame with the metrics
"""

def network_k_fold_validation(k, path = DATASET_PATH, model = None, invert=False):
    test_size = 1/k
    if invert:
        test_size = 1-test_size

    #metrics arrays
    conf_matrices = []
    sw_f = []
    ev_f = []
    shock_acc = []
    shock_rec = []
    shock_f = []

    dataset = pd.read_csv(path)

    for i in range(k):
        print(i+1 ,'/',k,' folds')
        #runs a complete training and test for the i-th fold
        start_index_i = int(dataset.count()[0]*i/k)
        pp_dataset_i = prp.get_timed_train_test(dataset, start_index = start_index_i, ordered = True)
        train_dataset_i = prp.get_train_test_sets(pp_dataset_i[0], pp_dataset_i[1], pp_dataset_i[2], pp_dataset_i[3])

        ANN, _ = nn.run_training(train_dataset_i, layers_sizes = LAYERS_SIZES, layers_activations = LAYERS_ACTIVATIONS, epochs_nb = EPOCHS_NB, batch_size = BATCH_SIZE, test_size = TEST_SIZE)
        timed_Xtest, timed_ytest = pp_dataset_i[1], pp_dataset_i[3]
        timed_ypred, raw_proba, true_variations, pred_variations, true_crossings = pop.get_prediction(dataset, ANN, timed_Xtest, timed_ytest)

        #compute the evaluation metrics
        conf_m, conf_m_norm = ev.get_confusion_matrices(timed_ytest['label'], timed_ypred['label'])
        acc = ev.accuracy_from_cm(conf_m)
        rec = ev.recall_from_cm(conf_m)
        f = ev.f_measure_from_cm(conf_m)

        #Stores the metrics values
        conf_matrices.append(conf_m_norm)
        sw_f.append(f[2])
        ev_f.append(f[0])
        shock_acc.append(acc[1])
        shock_rec.append(rec[1])
        shock_f.append(f[1])

    #Builds a dataframe
    results = pd.DataFrame()
    results['SW_class_f'] = sw_f
    results['EV_class_f'] = ev_f
    results['SH_class_acc'] = shock_acc
    results['SH_class_rec'] = shock_rec
    results['SH_class_f'] = shock_f

    #plot
    ev.graph_pred_from_var(true_variations, pred_variations, data_name = 'Fold n = ' + str(i))

    return conf_matrices, results


def create_dataset(shock_list_path = SHOCK_LIST_PATH, shock_nb = -1, name = 'default'):
    shock_list = pd.read_csv(shock_list_path, names = ['epoch'])
    dataset = pd.DataFrame()
    if(shock_nb == -1):
        for i in range(shock_list.count()[0]):
            shock_epoch = shock_list['epoch'].iloc[i]
            print('Loading sample ' + str(i + 1) + '/' + str(shock_list.count()[0]))
            shock_data = gather_and_predict_data(shock_epoch)
            dataset = pd.concat([dataset, shock_data], ignore_index = True)
    else:
        shock_list_indexes = [k for k in range(shock_list.count()[0])]
        for i in range(shock_nb):
            random_index = shock_list_indexes[rd.randint(0, len(shock_list_indexes) - 1)]
            shock_epoch = shock_list['epoch'].iloc[random_index]
            print('Loading sample ' + str(i + 1) + '/' + str(shock_nb))
            shock_data = gather_and_predict_data(shock_epoch)
            dataset = pd.concat([dataset, shock_data], ignore_index = True)
            shock_list_indexes.remove(random_index)
    final_dataset = acom.save_df(dataset, SAVE_PATH, name)
    return final_dataset

"""
Script used to train a mlp neural network easily
Inputs:
    pandas.DataFrame() data for training
    List               layers number of perceptrons
    List               layers activation function (ex: relu, tanh, softmax...)
    int                total number of epochs
    int                batch size
    str                mlp name
"""
def train_nn(dataset, layers_sizes = LAYERS_SIZES, layers_activations = LAYERS_ACTIVATIONS, epochs_nb = EPOCHS_NB, batch_size = BATCH_SIZE, test_size = TEST_SIZE, dropout = 0.0, name = 'last_trained'):
    timed_dataset = prp.get_timed_train_test(dataset)
    train_dataset = prp.get_train_test_sets(timed_dataset[0], timed_dataset[1], timed_dataset[2], timed_dataset[3])
    ANN, training = nn.run_training(train_dataset, layers_sizes, layers_activations, epochs_nb, batch_size, test_size)
    nn.save_model(MODEL_PATH + name + '.h5', ANN)
    return ANN, training

"""
"""
def pred_from_model(dataset, model):
    pp_dataset_i = prp.get_timed_train_test(dataset, test_size = 1.0, ordered = True)
    timed_Xtest, timed_ytest = pp_dataset_i[1], pp_dataset_i[3]
    timed_ypred, raw_proba, true_variations, pred_variations, true_crossings = pop.get_prediction(dataset, model, timed_Xtest, timed_ytest)
    # Plot result
    plt.plot(dataset['epoch'], dataset['label'], 'g-')
    plt.plot(timed_ypred['epoch'], timed_ypred['label'], 'r-')
    plt.show()
    return timed_ypred, raw_proba

"""
"""
def corrected_prediction(model, dataset, dt_corr, dt_density):
    scale_data = dataset.drop('label', axis = 1)
    unseen_data = dataset.drop('label', axis = 1)
    init_pred = nn.get_pred_timed(model, unseen_data, scale_data)
    proba = nn.get_prob_timed(model, unseen_data, scale_data)

    init_var = ev.get_var(init_pred)
    init_var = ev.get_category(init_var)

    corr_pred = pop.get_corrected_pred2(init_pred, proba, dt_corr)
    vcorr = ev.get_category(ev.get_var(corr_pred))
    vcorr = pop.corrected_var(vcorr, 15) #deletes variations faster than 15s
    corr_crossings = ev.crossings_from_var(vcorr)

    corr_pred = pop.crossings_density(corr_pred, corr_crossings, dt_density)
    final_crossings = pop.final_list(corr_pred)

    # Plot data
    plt.plot(dataset.index, dataset.label, 'g-')
    plt.plot(corr_pred.index, corr_pred.label, 'b-')
    plt.show()
    return corr_pred, vcorr, final_crossings

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                            UTILITY FUNCTIONS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
Function that gather and preprocess data for a single shock epoch
"""
def gather_and_predict_data(centered_epoch):
    shock_begin, shock_end = str(pd.Timestamp(centered_epoch) - pd.Timedelta('45m')).replace(' ', 'T'), str(pd.Timestamp(centered_epoch) + pd.Timedelta('45m')).replace(' ', 'T')
    shock_data = acom.download_multiparam_df(shock_begin, shock_end, PARAMETER_NAMES, PARAMETER_COLS)
    shock_data = pred.predict_file(shock_data)
    return shock_data


if __name__ == '__main__':
    print('scripts main')