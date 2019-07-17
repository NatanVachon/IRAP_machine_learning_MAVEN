# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:44:16 2019

@author: natan
"""

GRUSBECK_LIST_PATH = "../Data/datasets/MAVEN_shockGRUESBECK.txt"
FANG_LIST_PATH = "../Data/datasets/MAVEN_shockFANG.txt"

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score

import MAVEN_prediction as pred
import MAVEN_evaluation as ev
import MAVEN_scripts as S

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

def plot_trial(trial):
    losses = [-t["result"]["loss"] for t in trial.trials]
    X = [t["misc"]["vals"]["fl_neuron_nb"][0] for t in trial.trials]
    Y = [t["misc"]["vals"]["sl_neuron_nb"][0] for t in trial.trials]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, losses)
    ax.grid()
    plt.show()
    return

def plot_histories(histories, legends):
    plt.figure()
    for i in range(len(histories)):
        plt.plot(histories[i].history['acc'])
    plt.grid()
    plt.legend(legends, loc='upper left')
    plt.figure()
    for i in range(len(histories)):
        plt.plot(histories[i].history['loss'])
    plt.grid()
    plt.legend(legends, loc='lower left')

def test_lerp(manager, data, n):
    accs, recalls = [], []
    for i in range(n+1):
        acc, recall = ev.metrics_from_list(manager, data, 60, 10*60, lerp_delta=float(i)/n)
        accs.append(acc)
        recalls.append(recall)
    plot_data([i for i in range(n+1)], [accs, recalls], legend=[i for i in range(n+1)])
    return accs, recalls

def plot_dt_tol_metrics(manager, data, dts):
    accs, recalls = [], []
    for dt in dts:
        acc, recall = ev.metrics_from_list(manager, data, 3*60, dt)
        accs.append(acc)
        recalls.append(recall)
    plot_data([dt / 60 for dt in dts], [accs, recalls], ["Acc", "Recall"], xLabel="delta_t")
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

def get_begin_end(filepath, name):
    out = open("../Data/amda_data/" + name, "w")
    data = pd.read_csv(filepath)

    for i in range(len(data)):
        out.write(data.at[i, "begin"] + " " + data.at[i, "end"] + "\n")
    return

def create_timetable(filepath, name):
    out = open("../Data/amda_data/timetables/" + name, "w")
    data = pd.read_csv(filepath)

    for i in range(len(data)):
        t = str(pd.to_datetime(0.5 * (pd.Timestamp(data.at[i, "begin"]).timestamp() + pd.Timestamp(data.at[i, "end"]).timestamp()), unit='s'))
        out.write(t + " " + t + "\n")
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

def plot_acc_func_of_train_sample_nb(manager, data_train, data_test, percents):
    losses = []
    accuracies = []
    shock_nb = []
    n = len(data_train) / 100
    for i in percents:
        subset = data_train.loc[:int(i * n)]
        _ = S.train_nn(manager, subset)
        pred = manager.get_pred(data_train.drop("epoch", axis=1))
        acc, loss = accuracy_score(data_train["label"], pred["label"]), recall_score(data_train["label"], pred["label"], average="weighted")
        losses.append(loss)
        accuracies.append(acc)
        shock_nb.append(i)
    plot_data(shock_nb, [accuracies, losses], ["acc", "loss"])
    return losses, accuracies

def plot_acc_recall(accs, recalls):
    plt.figure()
    plt.plot(accs[1], recalls[1], 'ro')
    plt.plot(accs[0], recalls[0], 'bo')
    plt.xlabel("Accuracy")
    plt.ylabel("Recall")
    plt.title("K-Fold metrics with dt_tol = 5min")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.show()
    return

def recompute_probas():
    raw_data = pd.read_csv("../Data/datasets/MAVEN_V4_datasets/MAVEN_V4_FULL.txt")
    data = raw_data.copy().drop("label", axis=1)
    output_df = pd.DataFrame(columns=list(data.columns)+["proba_ev", "proba_sh", "proba_sw"])
    begin, end = 0, 0
    while True:
        begin = end
        end = next((i + 1 for i in range(begin + 1, len(data) - 1) if data.at[i + 1, "epoch"] - data.at[i, "epoch"] > 10), None)
        if end is None:
            break
        print("begin", begin, "end", end)
        temp_data = data.iloc[begin:end]
        temp_data.index = [i for i in range(len(temp_data))]
        temp_data = pred.predict_file(temp_data, continuous=True)
        output_df = pd.concat([output_df, temp_data], ignore_index=True)
    output_df.to_csv("MAVEN_FULL_proba.txt", encoding = 'utf-8', index = False)
    return

def clean_dataset():
    data = pd.read_csv("../Data/datasets/MAVEN_V4_datasets/MAVEN_V4_FULL.txt").drop("label", axis=1)
    output_df = pd.DataFrame()
    begin, end = 0, 0
    while end is not None:
        begin = end
        end = next((i + 1 for i in range(begin + 1, len(data) - 1) if data.at[i + 1, "epoch"] - data.at[i, "epoch"] > 10), None)
        print("begin", begin, "end", end)
        temp_data = data.iloc[begin:end]

        # Compute shock boundaries
        begin_shock, end_shock = pred.compute_boundary_indexes(temp_data)

        # Split data in three parts
        first_part = temp_data.iloc[:begin_shock].copy()
        last_part = temp_data.iloc[end_shock:].copy()

        # Compute direction
        direction = pred.compute_direction(first_part, last_part)

        if(direction != 1):
            output_df = pd.concat([output_df, temp_data], ignore_index=True)
    output_df.to_csv("MAVEN_FULL.txt", encoding = 'utf-8', index = False)
    return output_df

def convert_epochs(file, save_path):
    data = pd.read_csv(file)
    for i in range(len(data)):
        data.at[i, "epoch"] = pd.to_datetime(data.at[i, "epoch"], unit='s')
    data.to_csv(save_path, encoding="utf-8", index=False)
    return