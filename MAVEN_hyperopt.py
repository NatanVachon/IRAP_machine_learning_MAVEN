# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:43:46 2019

@author: natan
"""

import TrainingManagment as tm
import MAVEN_scripts as S
import MAVEN_evaluation as ev

from hyperopt import fmin, tpe, hp
import pandas as pd

data_train = pd.read_csv("../Data/datasets/MAVEN_V4_datasets/MAVEN_V4_train_1245689.txt").drop("SWIA_qual", axis=1)
data_test = pd.read_csv("../Data/datasets/MAVEN_V4_datasets/MAVEN_V4_700.txt").drop("SWIA_qual", axis=1)

#space = [hp.uniform('x', -10, 10), hp.uniform('y', -10, 10)]
#
#best = fmin(fn=lambda x: x[0] ** 2 + x[1] ** 2,
#    space=space,
#    algo=tpe.suggest,
#    max_evals=1000)
#print(best)

def global_optimization(fl_bounds, sl_bounds, max_evals=200):
    space = [hp.quniform('fl_neuron_nb', fl_bounds[0], fl_bounds[1], 1),
             hp.quniform('sl_neuron_nb', sl_bounds[0], sl_bounds[1], 1)]
    best = fmin(fn=fitness_function, space=space, algo=tpe.suggest, max_evals=max_evals)
    return best

def fitness_function(param):
    """
    param[0]: first hidden layer neuron nb
    param[1]: second hidden layer neuron nb
    """
    manager = tm.TrainingManager()
    manager["batch_size"] = 1024
    manager["epochs_nb"] = 35
    manager["layers_sizes"] = [8, int(param[0]), int(param[1]), 3]
    _ = S.train_nn(manager, data_train, verbose=0)
    acc, loss = ev.metrics_from_list(manager, data_test, 60, 5*60)

    return 2.0 * acc * loss / (acc + loss)
    # TODO compute cost