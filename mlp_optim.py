# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:22:23 2019

@author: natan
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import scripts as S

DATA_PATH = '../Data/datasets/MAVEN_last_30_ruptures.txt'

"""
This function is used to look for the best number of neurons in a 3 layes mlp
"""
def neuron_nb_opti(min_nb, max_nb, epochs_nb = 50, batch_size = 256, test_size = 0.15):
    histories = []
    dataset = pd.read_csv(DATA_PATH)
    # Trainings for different neuron nb in the hidden layer
    for i in range(min_nb, max_nb + 1):
        ANN, history = S.train_nn(dataset, [9, i, 3], ['relu', 'tanh', 'softmax'], epochs_nb, batch_size, test_size)
        histories.append(history)

    # Plot step
    plot_histories(histories, [i for i in range(min_nb, max_nb + 1)])
    return histories

def second_layer_neuron_nb_opti(min_nb, max_nb, epochs_nb = 200, batch_size = 128, test_size = 0.15):
    histories = []
    dataset = pd.read_csv(DATA_PATH)
    # Trainings for different neuron nb in the hidden layer
    for i in range(min_nb, max_nb + 1):
        ANN, history = S.train_nn(dataset, [9, 8, i, 3], ['relu', 'relu', 'sigmoid', 'softmax'], epochs_nb, batch_size, test_size)
        histories.append(history)

    # Plot step
    plot_histories(histories, [i for i in range(min_nb, max_nb + 1)])
    return histories

def activation_opti(epochs_nb = 50, batch_size = 256, test_size = 0.15):
    histories = []
    dataset = pd.read_csv(DATA_PATH)
    # relu hidden layer
    ANN, history = S.train_nn(dataset, [9, 8, 3], ['relu', 'relu', 'softmax'], epochs_nb, batch_size, test_size)
    histories.append(history)
    # tanh hidden layer
    ANN, history = S.train_nn(dataset, [9, 8, 3], ['relu', 'tanh', 'softmax'], epochs_nb, batch_size, test_size)
    histories.append(history)
    # sigmoid hidden layer
    ANN, history = S.train_nn(dataset, [9, 8, 3], ['relu', 'sigmoid', 'softmax'], epochs_nb, batch_size, test_size)
    histories.append(history)

    # Plot part
    plot_histories(histories, ['relu', 'tanh', 'sigmoid'])
    return histories


def plot_histories(histories, legends):
    plt.figure()
    for i in range(len(histories)):
#        plt.plot(np.log(1 - np.array(histories[i].history['acc'])))
        plt.plot(histories[i].history['acc'])
    plt.grid()
    plt.legend(legends, loc='upper left')
    plt.ylim(0.95, 1)

    plt.figure()
    for i in range(len(histories)):
#        plt.plot(np.log(histories[i].history['loss']))
        plt.plot(histories[i].history['loss'])
    plt.grid()
    plt.legend(legends, loc='lower left')
    plt.ylim(0, 0.1)

    plt.show()

if __name__ == '__main__':
    print('main optim')