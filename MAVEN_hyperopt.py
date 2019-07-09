# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:43:46 2019

@author: natan
"""

import TrainingManagment as tm
import preprocessing as prp
import mlp
import MAVEN_scripts as S

from hyperopt import fmin, tpe, hp, Trials
import pandas as pd
from sklearn.metrics import f1_score

#data = pd.read_csv("../Data/datasets/MAVEN_V4_datasets/MAVEN_V4_FULL.txt").drop("SWIA_qual", axis=1)
data = pd.read_csv("MAVEN_V4_FULL.txt").drop("SWIA_qual", axis=1)

def global_optimization(fl_bounds, sl_bounds, batch_size_bounds, max_evals=200):
    space = [hp.quniform('fl_neuron_nb', fl_bounds[0], fl_bounds[1], 1),
             hp.quniform('sl_neuron_nb', sl_bounds[0], sl_bounds[1], 1),
             2 ** hp.quniform('batch_power', batch_size_bounds[0], batch_size_bounds[1], 1)]
    trials = Trials()
    best = fmin(fn=fitness_function, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    return best, trials

def fitness_function(param):
    """
    param[0]: first hidden layer neuron nb
    param[1]: second hidden layer neuron nb
    param[2]: log2 of batch size
    """
    manager = tm.TrainingManager()
    manager["layers_sizes"] = [8, int(param[0]), int(param[1]), 3]
    manager["batch_size"] = int(param[2])
    manager["epochs_nb"] = 35

    timed_data = prp.split_data(data, test_size=0.2, ordered=True)
    train_dataset, manager.scaler = prp.scale_and_format(timed_data[0].drop("epoch", axis=1), timed_data[1].drop("epoch", axis=1), timed_data[2], timed_data[3])
    manager.model, history = mlp.run_training(train_dataset, layers_sizes=manager.params["layers_sizes"],layers_activations=manager.params["layers_activations"],
                                              epochs_nb=manager.params["epochs_nb"], batch_size=manager.params["batch_size"], verbose=0)

    pred = S.corrected_prediction(manager, timed_data[1], 70, plot=False)
    score = f1_score(timed_data[3]["label"], pred["label"], average="weighted")
    return -score