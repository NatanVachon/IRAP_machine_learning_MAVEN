# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:44:16 2019

@author: natan
"""

GRUSBECK_LIST_PATH = "../Data/datasets/MAVEN_shockGRUESBECK.txt"
FANG_LIST_PATH = "../Data/datasets/MAVEN_shockFANG.txt"

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model

import MAVEN_communication_AMDA as acom
import MAVEN_prediction as pred
import MAVEN_evaluation as ev
import MAVEN_scripts as S
import TrainingManagment as tm
import numpy as np

def load_dataset(n):
    data = pd.read_csv("../Data/datasets/MAVEN_V4_datasets/MAVEN_V4_" + str(n) + "00.txt")
    data = data.drop("SWIA_qual", axis=1)
    return data

def save_epoch_label(data, filepath):
    file = open(filepath, 'w')
    for i in range(len(data)):
        file.write(str(pd.to_datetime(data.at[i, 'epoch'], unit='s')).replace(' ', 'T') + ' ')
        file.write(str(data.at[i, 'label']) + "\n")
    file.close()
    return

def unique_time_event(dt = 40 * 60, grusbeck_list = GRUSBECK_LIST_PATH, fang_list = FANG_LIST_PATH):
    f_gru = open(grusbeck_list, 'r')
    f_fang = open(fang_list, 'r')
    # Parse shocks epochs
    gru_epochs = []
    fang_epochs = []
    for i, l in enumerate(f_gru):
        words = l.split(' ')
        gru_epochs.append(pd.Timestamp(year=int(words[0]), month=int(words[1]), day=int(words[2]), hour=int(words[3]), minute=int(words[4]), second=int(words[5][:1])))
    for i, l in enumerate(f_fang):
        words = l.split(' ')
        fang_epochs.append(pd.Timestamp(year=int(words[0]), month=int(words[1]), day=int(words[2]), hour=int(words[3]), minute=int(words[4]), second=int(words[5][:1])))

    # Get common shocks
    common_shocks = []
    gru_common = []
    fang_common = []
    for i, g in enumerate(gru_epochs):
        fang_common_epoch = next((f for i, f in enumerate(fang_epochs) if abs(g.timestamp() - f.timestamp()) < dt), None)
        if fang_common_epoch is not None:
            common_shocks.append(str(pd.to_datetime(0.5 * (fang_common_epoch.timestamp() + g.timestamp()), unit='s').strftime("%Y-%m-%dT%H:%M:%S")))
            gru_common.append(g.strftime("%Y-%m-%dT%H:%M:%S"))
            fang_common.append(fang_common_epoch.strftime("%Y-%m-%dT%H:%M:%S"))
    return common_shocks, gru_common, fang_common

def rupture_detection_comparison(common_shocks, gru_common, fang_common, sample_nb):
    ruptures_epochs = []
    for i in range(sample_nb):
        print("Sample " + str(i + 1) + '/' + str(sample_nb))
        shock_begin, shock_end = str(pd.Timestamp(common_shocks[i]) - pd.Timedelta('20m')).replace(' ', 'T'), str(pd.Timestamp(common_shocks[i]) + pd.Timedelta('20m')).replace(' ', 'T')
        shock_data = acom.download_multiparam_df(shock_begin, shock_end)
        shock_epoch = shock_data.at[int(pred.compute_shock_position(shock_data)), 'epoch']
        ruptures_epochs.append(shock_epoch)
    # Compute deltas
    gru_deltas = []
    fang_deltas = []
    for i in range(sample_nb):
        gru_deltas.append(pd.Timestamp(gru_common[i]).timestamp() - pd.Timestamp(ruptures_epochs[i]).timestamp())
        fang_deltas.append(pd.Timestamp(fang_common[i]).timestamp() - pd.Timestamp(ruptures_epochs[i]).timestamp())
    # Plot distribution
    plot_deltas_distribution(gru_deltas, fang_deltas)
    return ruptures_epochs, gru_deltas, fang_deltas

def plot_deltas_distribution(gru_deltas, fang_deltas):
    sns.distplot(gru_deltas, label="Grusbeck")
    sns.distplot(fang_deltas, label="Fang")
    plt.legend()
    plt.grid()
    return

def get_shock_epochs(data):
    var = ev.get_var(data)
    var = ev.get_category(var)
    shocks = []
    for i in range(var.count()[0]):
        if(var.at[i, "follow_class"] == 1):
            shocks.append(pd.to_datetime(var.at[i, "epoch"], unit='s'))
    return shocks

def metrics_over_dt(ANN, data, delta_t_list):
    accs, recalls = [], []
    for i, delta_t in enumerate(delta_t_list):
        acc, recall = S.test(ANN, data, 60, delta_t)
        accs.append(acc)
        recalls.append(recall)

    return accs, recalls

def plot_data(x, y, legend=[], shapes=None, xLabel="", yLabel=""):
    plt.figure()
    if shapes is None:
        for i in range(len(y)):
            plt.plot(x, y[i])
    else:
        for i in range(len(y)):
            plt.plot(x, y[i], shapes[i])
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.legend(legend)
    plt.grid()
    plt.show()

def test_lerp(manager, data, n):
    accs, recalls = [], []
    for i in range(n+1):
        acc, recall = ev.metrics_from_list(manager, data, 60, 10*60, lerp_delta=float(i)/n)
        accs.append(acc)
        recalls.append(recall)
    plot_data([i for i in range(n+1)], [accs, recalls], legend=[i for i in range(n+1)])
    return accs, recalls

def dt_corr_optim(manager, data, dts):
    accs, recalls = [], []
    for dt in dts:
        acc, recall = ev.metrics_from_list(manager, data, dt, 5*60)
        accs.append(acc)
        recalls.append(recall)
    plot_data(dts, [accs, recalls], [str(dt) for dt in dts])
    return accs, recalls

def duplicate_epochs(filepath):
    f = open(filepath, "r")
    fd = open("new.txt", "w")
    lines = f.readlines()
    for i in range(len(lines)):
        line = lines[i][:-1]
        fd.write(line + ' ' + line + '\n')
    f.close()
    fd.close()
    return

def plot_model_shape(model, filepath):
    plot_model(model, to_file=filepath, show_shapes=True, show_layer_names=True)
    return

def plot_histograms(datas):
    """Plots histograms of data distributions"""
    n = len(datas)
    for column in datas[0]:
        if column not in ["label", "epoch", "SWIA_qual"]:
            print(column)
            plt.figure()
            for data in datas:
                plt.hist(data[column], bins=50, alpha=1.0/n)
            plt.legend([str(i) for i in range(n)])
            plt.show()
    return

def plot_acc_func_of_train_sample_nb(data_train, data_test, percents):
    manager = tm.TrainingManager()
    manager["epochs_nb"] = 30
    manager["batch_size"] = 64
    recalls = []
    accuracies = []
    shock_nb = []
    n = data_train.count()[0] / 100
    for i in percents:
        subset = data_train.loc[:i * n]
        _ = S.train_nn(manager, subset)
        acc, recall = ev.metrics_from_list(manager, data_test, 60, 5*60)
        recalls.append(recall)
        accuracies.append(acc)
        shock_nb.append(i)
    plot_data(shock_nb, [accuracies, recalls], ["acc", "recalls"])
    return recalls, accuracies, shock_nb

def co_learning_matrix(I, J):
    manager = tm.TrainingManager()
    manager["batch_size"] = 256
    manager["epochs_nb"] = 50
    manager["layers_sizes"] = [8, 7, 5, 3]
    A, R = np.zeros((max(I), max(J))), np.zeros((max(I), max(J)))
    for i in I:
        data_train = load_dataset(i)
        for j in J:
            data_test = load_dataset(j)
            _ = S.train_nn(manager, data_train)
            acc, recall = ev.metrics_from_list(manager, data_test, 60, 5*60)
            A[i-1, j-1] = acc
            R[i-1, j-1] = recall
    return A, R