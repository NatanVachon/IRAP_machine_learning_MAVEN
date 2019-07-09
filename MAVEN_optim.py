# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:10:37 2019

@author: natan
"""

import TrainingManagment as tm
import pickle as pkl
import pandas as pd

manager = tm.TrainingManager()
manager.name = "MAVEN_opti_3"
manager["layers_sizes"] = [8, 40, 38, 3]
manager["batch_size"] = 512
manager["epochs_nb"] = 10 #400

#data = pd.read_csv("MAVEN_FULL.txt").drop("epoch", axis=1)
data = pd.read_csv("../Data/datasets/MAVEN_V4_datasets/MAVEN_FULL.txt").drop("epoch", axis=1)

manager.run_training(data, weight=True)

del data

# Save data
g = open("result.pkl", "wb")
pkl.dump(manager, g, pkl.HIGHEST_PROTOCOL)
g.close()