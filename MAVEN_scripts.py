# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:18:13 2019

This file containts scripts
@author: natan
"""

import pandas as pd
import random as rd
import matplotlib.pyplot as plt

import MAVEN_postprocessing as pop
import MAVEN_evaluation as ev
import MAVEN_communication_AMDA as acom
import MAVEN_prediction as pred

SAVE_PATH = "d:/natan/Documents/IRAP/Data/datasets/"
#SHOCK_LIST_PATH = '../Data/datasets/ShockMAVEN_dt1h_list.txt'
SHOCK_LIST_PATH = '../Data/datasets/ShockMAVEN_list.txt'

DATASET_PATH = '../Data/datasets/simple_dataset.txt'

def create_dataset(shock_list_path = SHOCK_LIST_PATH, shock_nb = -1, random = True, offset_index = 0, name = 'default', plot = False):
    shock_list = pd.read_csv(shock_list_path)
    if shock_nb == -1:
        shock_nb = shock_list.count()[0]
    dataset = pd.DataFrame()
    if random:
        shock_list_indexes = [k for k in range(shock_list.count()[0])]
        for i in range(shock_nb):
            random_index = shock_list_indexes[rd.randint(0, len(shock_list_indexes) - 1)]
            shock_epoch = shock_list.at[random_index, 'epoch']
            print('Loading sample ' + str(i + 1) + '/' + str(shock_nb))
            print("epoch: " + str(shock_epoch))
            shock_data = gather_and_predict_data(shock_epoch, plot)
            dataset = pd.concat([dataset, shock_data], ignore_index = True)
            shock_list_indexes.remove(random_index)
    else:
        for i in range(offset_index, offset_index + shock_nb):
            shock_epoch = shock_list.at[i, 'epoch']
            print('Loading sample ' + str(i - offset_index + 1) + '/' + str(shock_nb))
            shock_data = gather_and_predict_data(shock_epoch, plot)
            dataset = pd.concat([dataset, shock_data], ignore_index = True)
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
def train_nn(manager, dataset, verbose=1):
    history = manager.run_training(dataset.drop("epoch", axis=1), verbose=verbose)
    return history

"""
"""
def test(manager, data, dt_corr):
    pred_data = corrected_prediction(manager, data, dt_corr, plot=False)
    manager.cm, _ = ev.get_confusion_matrices(data["label"], pred_data["label"])
    return manager.cm
"""
"""
def corrected_prediction(manager, dataset, dt_corr=70, plot=True):
    if 'label' in dataset.columns:
        proba = manager.get_prob(dataset.drop(["epoch", "label"], axis=1))
    else:
        proba = manager.get_prob(dataset.drop("epoch", axis=1))
    proba["epoch"] = dataset["epoch"]
    corr_pred = pop.get_corrected_pred(proba, dt_corr)

    if plot:
        # Plot data
        if 'label' in dataset.columns:
            plt.plot(dataset.index, dataset.label, 'g-')
        plt.plot(corr_pred.index, corr_pred.label, 'b-')
        if 'label' in dataset.columns:
            plt.legend(['True label', 'Predicted label'], loc = 'upper right')
        else:
            plt.legend(['Predicted label'], loc = 'upper right')
        plt.show()
    return corr_pred #,vcorr, final_crossings

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                            UTILITY FUNCTIONS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""
Function that gather and preprocess data for a single shock epoch
"""
def gather_and_predict_data(centered_epoch, plot = False):
    shock_begin, shock_end = str(pd.Timestamp(centered_epoch) - pd.Timedelta('20m')).replace(' ', 'T'), str(pd.Timestamp(centered_epoch) + pd.Timedelta('20m')).replace(' ', 'T')
    shock_data = acom.download_multiparam_df(shock_begin, shock_end, acom.PARAMETER_NAMES, acom.PARAMETER_COLS)
    shock_data = pred.predict_file(shock_data, plot)
    return shock_data

def plot_histories(histories, legends):
    if isinstance(histories, list):
        # Accuracy
        plt.figure()
        for i in range(len(histories)):
            plt.plot(histories[i].history['acc'])
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.grid()
        plt.legend(legends, loc='lower right')
        # Loss
        plt.figure()
        for i in range(len(histories)):
            plt.plot(histories[i].history['loss'])
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.grid()
        plt.legend(legends, loc='upper right')

        plt.show()
        return
    else:
        # Accuracy
        plt.figure()
        plt.plot(histories.history['acc'])
        plt.legend([legends], loc='lower right')
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.grid()
        # Loss
        plt.figure()
        plt.plot(histories.history['loss'])
        plt.legend([legends], loc='upper right')
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.grid()

        plt.show()
        return



if __name__ == '__main__':
    print('scripts main')