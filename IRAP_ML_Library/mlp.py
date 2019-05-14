# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:22:59 2019

This file contains every functions related to multi layers perceptrons handling and training
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                       IMPORTS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import keras.backend as B
from tensorflow import keras as ks

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                   DEFAULT PARAMETERS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
FEATURE_NB = 8
CLASS_NB = 3
EPOCHS_NB = 200
BATCH_SIZE = 256
TEST_SIZE = 0.3
LAYERS_SIZES = [FEATURE_NB, FEATURE_NB, CLASS_NB, CLASS_NB]
LAYERS_ACTIVATIONS = ['relu', 'relu', 'tanh', 'softmax']

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                   FUNCTIONS DEFINITION
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def run_training(datasets, layers_sizes = LAYERS_SIZES, layers_activations = LAYERS_ACTIVATIONS, epochs_nb = EPOCHS_NB,
                 batch_size = BATCH_SIZE, test_size = TEST_SIZE, dropout = 0.0):
    """
    Function training a neural network according to some parameters and dataset

    Inputs:
        pandas.DataFrame()[] List of: X_train, X_test, y_train, y_test (preprocessed)
        int[]                List of layers sizes
        activation[]         List of layers activation
        int                  Number of epoch
        int                  Batch size
        float                Test proportion (between 0 and 1)

    Returns: trained mlp, training history
    """
    ANN = create_model(layers_sizes, layers_activations, dropout = dropout)
    training = compile_and_fit(ANN, datasets[0], datasets[2], epochs_nb, batch_size)
    return ANN, training


def create_model(lay_s, act, dropout=0.0):
    """
    Creates a keras model
    Arguments:
        - lay_s : ex : layer_sizes = [11,11,6] will give an ANN with layers (11, 11, 6)
        - act : ex : act = ['relu', 'relu', 'softmax']
        - dropout : dropout proportion, default to 0 (applied between every layer)

    Returns: Created model
    """
    #initializing the model
    model = Sequential()
    #adding the input layer
    model.add(Dense(lay_s[0], activation = act[0], input_shape=(lay_s[0],)))
    if dropout > 0:
        model.add(Dropout(dropout))
    #adding the other layers
    for i in range(1,len(lay_s)):
        model.add(Dense(lay_s[i], activation = act[i]))
        if dropout > 0:
            model.add(Dropout(dropout))
    return model


def jaccard_distance(y_true, y_pred, smooth=100):
    """
    Explication du Jaccard
    Les ensembles X et Y à considérer sont:
        Xc : éléments de classe c dans les prédictions
        Yc : éléments de classe c dans le set de test

    Pour la précision, les ensembles considérés sont différents:
        Xc : éléments de classe c dans les prédictions
        Y : ensemble des éléments du set de test

    Dans les 2 cas on somme ensuite sur l'ensemble des classes c

    A noter que le jaccard et la précision sont donc identiques dans le cas d'une classification binaire
    """
    intersection = B.sum(B.abs(y_true * y_pred), axis=-1)
    sum_ = B.sum(B.abs(y_true) + B.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def compile_and_fit(model, X_train, y_train, n_epochs, b_s, val_size=0, loss_name = jaccard_distance):
    """
    Arguments:
        - model : model to compile and train
        - X_train : train set to feed to the network
        - y_train : labels corresponding to the X_train data set
        - n_epochs : number of epochs of training
        - b_s : batch size during the training
        - loss_name : loss to use (default: jaccard_distance)
    Returns:
        training is the history of the training
    """
    model.compile(optimizer = 'adam', loss = [loss_name], metrics = ['acc'])
    training = model.fit(X_train, y_train, validation_split = val_size, epochs = n_epochs, batch_size = b_s, verbose = 1)
    return training


def get_prob(model, scaler, X_test):
    """
    Get the probability with which the model predicted each class
    """
    X_test = scaler.transform(X_test)
    y_prob = model.predict(X_test)
    return y_prob

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                   UTILITY FUNCTIONS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def load_model(path):
    """
    Loads an existing model
    """
    model = ks.models.load_model(path, custom_objects={'jaccard_distance': jaccard_distance})
    return model