# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:06:01 2019
This is the main file used to launch every function in separated files

@author: natan
"""

import communication_AMDA as acom
import prediction as pred
import preprocessing as prp
import neural_networks as nn
import pandas as pd
import random as rd

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                            PROJECT CONSTANTS DEFINITION
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# PATHS
SHOCK_LIST_PATH = '../Data/datasets/ShockMAVEN_list.txt'
DATASET_SAVE_PATH = '../Data/datasets/MAVEN_reduced_dataset.txt'
DATASET_PATH = '../Data/datasets/MAVEN_dataset_50.txt'

# MEASURES PARAMETERS
PARAMETER_NAMES = ["ws_0", "ws_1", "mav_bkp_mso", "ws_2", "ws_3", "mav_swiakp_vmso"]
PARAMETER_COLS = [["epoch", "rho"], ["epoch", "deriv_r"], ["epoch", "mag_x", "mag_y", "mag_z"], ["epoch", "totels_1"], ["epoch", "totels_8"], ["epoch", "SWIA_vel_x", "SWIA_vel_y", "SWIA_vel_z"]]



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                                   MAIN FUNCTION
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if __name__ == "__main__":
    print('main main')