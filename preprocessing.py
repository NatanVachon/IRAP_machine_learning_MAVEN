# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:12:13 2019

Preprocessing file. The following functions are used to prepare datasets to be
used for training Neural Networks

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import np_utils

"""
Getting the train and test sets
Arguments:
    timed_data : a given pandas.DataFrame() to convert to test and train sets, containing at least a 'label' and an 'epoch' columns
    ordered : defines if the data sets created are shuffled (default: ordered=False) or chronological (ordered=True).
    test_size : proportion of samples included in the test set (default: 0.3)

    nb_class : defines if the problem has to be downgraded to 3 classes instead of 4

Return:
    X_train_timed, y_train_timed, X_test_timed, y_test_timed
    (These variables are called '_timed' because they still contain the 'epoch' information of the initial data)
"""
def get_timed_train_test(timed_data, test_size = 0.3, nb_class = 3, ordered = False, start_index = 0):

    ###########################
    #this part is specific to the martian problem
    ###########################
    if nb_class == 3 :
        labels = np.ravel(timed_data['label'])
        timed_data['label'] = labels
    elif nb_class != 4 :
        print("WARNING: wrong number of classes")
        return None
    ############################
    y = pd.DataFrame()
    y['epoch'] = timed_data['epoch']
    y['label'] = timed_data['label']

    X = timed_data.copy()
    del X['label']

    if ordered :
        if start_index == 0:
            split_index = int(test_size*timed_data.count()[0])

            X_test_timed = X.iloc[0:split_index,:]
            y_test_timed = y.iloc[0:split_index,:]

            X_train_timed = X.iloc[split_index:,:]
            y_train_timed = y.iloc[split_index:,:]

        else :
            split_index1 = start_index
            split_index2 = start_index + int(test_size*timed_data.count()[0])

            X_test_timed = X.iloc[split_index1:split_index2,:]
            y_test_timed = y.iloc[split_index1:split_index2,:]

            X_train_timed = pd.concat([X.iloc[0:split_index1,:], X.iloc[split_index2:,:]], ignore_index=True)
            y_train_timed = pd.concat([y.iloc[0:split_index1,:], y.iloc[split_index2:,:]], ignore_index=True)

    else:
        X_train_timed, X_test_timed, y_train_timed, y_test_timed = train_test_split(X,y,test_size = test_size)

    return X_train_timed, X_test_timed, y_train_timed, y_test_timed

"""
Getting the train and test sets without any time info and apply the scaling here
"""
def get_train_test_sets(X_train_timed, X_test_timed, y_train_timed, y_test_timed):
    X_train = X_train_timed.copy()
    X_test = X_test_timed.copy()

    y_train = one_hot_encode(y_train_timed['label'].tolist())
    y_test =  one_hot_encode(y_test_timed['label'].tolist())

    del X_train['epoch']
    del X_test['epoch']

    scaler = StandardScaler().fit(X_train)
    # Scale the train and test set
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

"""
Useful functions to switch between labels representations
    The 2 different representations are represented here for an example with 3 distinct classes

    labels = [0, 0, 1, 2, 1, 0, 2, 2, 0]

    y = [[1, 0, 0],
         [1, 0, 0],
         [0, 1, 0],
         [0, 0, 1],
         [0, 1, 0],
         [1, 0, 0],
         [0, 0, 1],
         [0, 0, 1],
         [1, 0, 0]]

"""
from sklearn.preprocessing import LabelEncoder

def one_hot_encode(labels):
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded = encoder.transform(labels)
    # convert integers to dummy variables (i.e. one hot encoded)
    y = np_utils.to_categorical(encoded)
    return y