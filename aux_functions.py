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

import communication_AMDA as acom
import prediction as pred

def save_file(data, filepath):
    file = open(filepath, 'w')
    for i in range(len(data)):
        file.write(str(data.at[i, 'epoch'] + ' '))
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
