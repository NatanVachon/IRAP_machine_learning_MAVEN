# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:06:01 2019
This is the main file used to launch every function in separated files

@author: natan
"""

import communication_AMDA as acom
import prediction as pred
import preprocessing as pp
import neural_networks as nn
import pandas as pd
import random as rd

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                            PROJECT CONSTANTS DEFINITION
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# PATHS
SHOCK_LIST_PATH = '../Data/datasets/ShockMAVEN_list.txt'
DATASET_SAVE_PATH = '../Data/datasets/MAVEN_reduced_dataset.txt'
DATASET_PATH = '../Data/datasets/MAVEN_reduced_dataset.txt'
LOAD_MODEL_PATH = '../Data/models/last_model.h5'
SAVE_MODEL_PATH = '../Data/models/last_model.h5'

# MEASURES PARAMETERS
PARAMETER_NAMES = ["ws_0", "ws_1", "mav_bkp_mso", "ws_2", "ws_3", "mav_swiakp_vmso"]
PARAMETER_COLS = [["epoch", "rho"], ["epoch", "deriv_r"], ["epoch", "mag_x", "mag_y", "mag_z"], ["epoch", "totels_1"], ["epoch", "totels_8"], ["epoch", "SWIA_vel_x", "SWIA_vel_y", "SWIA_vel_z"]]

# TRAINING PARAMETERS
TRAIN_FROM_EXISTING = False

FEATURE_NB = 10
CLASS_NB = 3
EPOCHS_NB = 3
BATCH_SIZE = 256
TEST_SIZE = 0.3

LAYERS_SIZES = [FEATURE_NB, 66, 22, CLASS_NB]
LAYERS_ACTIVATIONS = ['relu', 'relu', 'relu', 'softmax']

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                            DATASET CREATION
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""
Function that creates a csv file containing data centered around the shock (-+ 30min)
epoch written in the shock list.
"""
def create_dataset(shock_list_path = SHOCK_LIST_PATH, shock_nb = -1):
    shock_list = pd.read_csv(shock_list_path, names = ['epoch'])
    dataset = pd.DataFrame()
    if(shock_nb == -1):
        for i in range(shock_list.count()[0]):
            shock_epoch = shock_list['epoch'].iloc[i]
            print('Loading sample ' + str(i + 1) + '/' + str(shock_list.count()[0]))
            shock_data = gather_and_preprocess_data(shock_epoch)
            dataset = pd.concat([dataset, shock_data], ignore_index = True)
    else:
        shock_list_indexes = [k for k in range(shock_list.count()[0])]
        for i in range(shock_nb):
            random_index = shock_list_indexes[rd.randint(0, len(shock_list_indexes))]
            shock_epoch = shock_list['epoch'].iloc[random_index]
            print('Loading sample ' + str(i + 1) + '/' + str(shock_nb))
            shock_data = gather_and_preprocess_data(shock_epoch)
            dataset = pd.concat([dataset, shock_data], ignore_index = True)
            shock_list_indexes.remove(random_index)
    return dataset

"""
Function that gather and preprocess data for a single shock epoch
"""
def gather_and_preprocess_data(centered_epoch):
    shock_begin, shock_end = str(pd.Timestamp(centered_epoch) - pd.Timedelta('45m')).replace(' ', 'T'), str(pd.Timestamp(centered_epoch) + pd.Timedelta('45m')).replace(' ', 'T')
    shock_data = acom.download_multiparam_df(shock_begin, shock_end, PARAMETER_NAMES, PARAMETER_COLS)
    pred.plot_variables(shock_data) #TODO: REMOVE
    shock_data = pred.predict_file(shock_data)
    return shock_data

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                            DATASET PREPROCESSING
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""
Function preprocessing a dataset so it can be used to train a NN
TODO: Maybe put this in preprocessing.py

Inputs:
    pandas.DataFrame() Timed dataset
    float              Test size (between 0.0 and 1.0), defalut = 0.3
    int                Class number, default = 3 CLASS NB != 3 NOT IMPLEMENTED YET
"""
def preprocess_dataset(timed_dataset, test_size = 0.3, class_nb = 3):
    X_train_timed, X_test_timed, y_train_timed, y_test_timed = pp.get_timed_train_test(timed_dataset, test_size, class_nb)
    X_train, X_test, y_train, y_test = pp.get_train_test_sets(X_train_timed, X_test_timed, y_train_timed, y_test_timed)
    return X_train, X_test, y_train, y_test

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                           NEURAL NETWORK TRAINING
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""
Function training a neural network according to some parameters and dataset

Inputs:
    pandas.DataFrame()[] List of: X_train, X_test, y_train, y_test (preprocessed)
    int[]                List of layers sizes
    activation[]         List of layers activation
    int                  Number of epoch
    int                  Batch size
    float                Test proportion (between 0 and 1)
"""
def run_training(datasets, layers_sizes = LAYERS_SIZES, layers_activations = LAYERS_ACTIVATIONS, epochs_nb = EPOCHS_NB, batch_size = BATCH_SIZE, test_size = TEST_SIZE):
    if TRAIN_FROM_EXISTING:
        ANN = nn.load_model(LOAD_MODEL_PATH)
    else:
        ANN = nn.create_model(layers_sizes, layers_activations)
    nn.compile_and_fit(ANN, datasets[0], datasets[2], epochs_nb, batch_size)
    nn.save_model(SAVE_MODEL_PATH, ANN)
    return ANN

"""
Main function used to create a dataset file by parsing data from AMDA.
Output data is nan free and labeled.
"""
if __name__ == "__main__":
    dataset = create_dataset(shock_nb = 10)
    pp_dataset = preprocess_dataset(dataset)
    ANN = run_training(pp_dataset)