# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:22:23 2019

@author: natan
"""
import pandas as pd
import matplotlib.pyplot as plt

import MAVEN_scripts as S
import MAVEN_neural_networks as nn

DATA_PATH = '../Data/datasets/MAVEN_V4_FULL.txt'

"""
This function is used to look for the best number of neurons in a 3 layes mlp
"""
def neuron_nb_opti(min_nb, max_nb, epochs_nb = 50, batch_size = 256, test_size = 0.15, dataset=None):
    histories = []
    if dataset is None:
        dataset = pd.read_csv(DATA_PATH)
    # Trainings for different neuron nb in the hidden layer
    for i in range(min_nb, max_nb + 1):
        ANN, history = S.train_nn(dataset, [8, i, 3], ['relu', 'tanh', 'softmax'], epochs_nb, batch_size, test_size)
        histories.append(history)

    # Plot step
    plot_histories(histories, [i for i in range(min_nb, max_nb + 1)])
    return histories

def second_layer_neuron_nb_opti(min_nb, max_nb, epochs_nb = 200, batch_size = 128, test_size = 0.15, dataset=None):
    histories = []
    if dataset is None:
        dataset = pd.read_csv(DATA_PATH)
    # Trainings for different neuron nb in the hidden layer
    for i in range(min_nb, max_nb + 1):
        ANN, history = S.train_nn(dataset, [8, 8, i, 3], ['relu', 'relu', 'sigmoid', 'softmax'], epochs_nb, batch_size, test_size)
        histories.append(history)

    # Plot step
    plot_histories(histories, [i for i in range(min_nb, max_nb + 1)])
    return histories

def two_hl_neuron_nb_opti(first_layer_nb, second_layer_nb, epochs_nb = 200, batch_size = 512, test_size = 0.2, recall = 0.15, dataset=None):
    histories = []
    if dataset is None:
        dataset = pd.read_csv(DATA_PATH)
    for i,j in zip(first_layer_nb, second_layer_nb):
        ANN, history = S.train_nn(dataset, [nn.FEATURE_NB, i, j, nn.CLASS_NB], ['relu', 'relu', 'tanh', 'softmax'], epochs_nb, batch_size, test_size)
        histories.append(history)
    plot_histories(histories, list(zip(first_layer_nb, second_layer_nb)))
    return histories

def activation_opti(epochs_nb = 50, batch_size = 256, test_size = 0.15, dataset=None):
    histories = []
    if dataset is None:
        dataset = pd.read_csv(DATA_PATH)
    # relu hidden layer
    ANN, history = S.train_nn(dataset, [8, 6, 3], ['relu', 'relu', 'softmax'], epochs_nb, batch_size, test_size)
    histories.append(history)
    # tanh hidden layer
    ANN, history = S.train_nn(dataset, [8, 6, 3], ['relu', 'tanh', 'softmax'], epochs_nb, batch_size, test_size)
    histories.append(history)
    # sigmoid hidden layer
    ANN, history = S.train_nn(dataset, [8, 6, 3], ['relu', 'sigmoid', 'softmax'], epochs_nb, batch_size, test_size)
    histories.append(history)

    # Plot part
    plot_histories(histories, ['relu', 'tanh', 'sigmoid'])
    return histories

def batch_size_opti(batch_sizes, dataset=None):
    histories = []
    if dataset is None:
        dataset = pd.read_csv(DATA_PATH)
    if "SWIA_qual" in dataset.columns:
        dataset = dataset.drop(["SWIA_qual"], axis=1)
    for i in range(len(batch_sizes)):
        _, history = S.train_nn(dataset, batch_size = batch_sizes[i])
        histories.append(history)
    plot_histories(histories, batch_sizes)
    return histories

def plot_histories(histories, legends):
    plt.figure()
    for i in range(len(histories)):
        plt.plot(histories[i].history['acc'])
    plt.grid()
    plt.legend(legends, loc='upper left')
    #plt.ylim(0.95, 1)

    plt.figure()
    for i in range(len(histories)):
        plt.plot(histories[i].history['loss'])
    plt.grid()
    plt.legend(legends, loc='lower left')
    #plt.ylim(0, 0.1)

    plt.show()

if __name__ == '__main__':
    print('main optim')