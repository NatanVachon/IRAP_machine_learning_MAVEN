# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:18:17 2019

This file contains everything about predict labels on raw data in order to
train NNs.

"""

import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt

PRED_VECTOR = [ -1/50.0, -5e-4, 1, 1 ] # Outbound defined: SWIA_x, totels1, deriv_r first, deriv_r last
PRED_VAR_DIAG = [ 1, 2e-4, 1, 1 ]

"""
Detection parameters
"""
MOVING_AVERAGE_NB = 19

#SHOCK_TOTELS_THRESHOLD = 5e4 # nb of shocks
SHOCK_TOTELS_THRESHOLD_RATIO = 0.2
SOLAR_WIND_DELTA_THRESHOLD = 50 # km/s

"""
Semi-automatic prediction of labels for every timestamp in the dataframe.
To determine the label we assume that in an hour we can detect one cross

Arguments:
    pandas.DataFrame() containing data for several timestamps

Returns:
    Input dataframe with an additional column containing labels if the direction
    is detected, else return None
"""
def predict_file(data, plot = False):
    if "label" in data.columns:
        data = data.drop(["label"], axis=1)
    # Compute shock boundaries
    begin_shock, end_shock = compute_boundary_indexes(data)
    # Fill nan values with median values per class
    data = fill_nan_values(data, begin_shock, end_shock)
    if plot:
        plot_variables(data, begin_shock, end_shock)
    # Split data in three parts
    first_part = data.iloc[:begin_shock].copy()
    last_part = data.iloc[end_shock:].copy()
    # Compute direction
    direction = compute_direction(first_part, last_part)
    # Check for label detection
    if direction == 1:
        print("Direction not found")
        return None
    # Assign label
    data['label'] = [0 for i in range(data.count(0)[0])]
    data['label'].iloc[:begin_shock] = [direction for i in range(begin_shock)]
    data['label'].iloc[begin_shock:end_shock] = [1 for i in range(end_shock - begin_shock)]
    data['label'].iloc[end_shock:] = [(2 - direction) for i in range(len(data['label'].iloc[end_shock:]))]
    return data

"""
Replaces nan values with median value
Input:
    pandas.DataFrame()

Returns:
    pandas.DataFrame() with no more nan values
"""
def fill_nan_values(data, begin_shock, end_shock):
    for i in range(1, data.count(1)[0]):
        data.iloc[:begin_shock,i] = data.iloc[:begin_shock,i].fillna(data.iloc[:begin_shock,i].median())
        data.iloc[begin_shock:end_shock,i] = data.iloc[begin_shock:end_shock,i].fillna(data.iloc[begin_shock:end_shock,i].median())
        data.iloc[end_shock:,i] = data.iloc[end_shock:,i].fillna(data.iloc[end_shock:,i].median())
    return data

"""
Determines if the satellite is ascending or descending

Arguments:
    pandas.DataFrame(), First period data
    pandas.DataFrame(), Last period data

Returns:
    0 if ascending
    1 if not sure
    2 if descending
"""
def compute_direction(first_part, last_part):
    # Ion velocity
    mean_sw_vel_1 = np.mean(first_part["SWIA_vel_x"])
    #print("mean_sw_vel_1: " + str(mean_sw_vel_1))
    mean_sw_vel_2 = np.mean(last_part["SWIA_vel_x"])
    #print("mean_sw_vel_2: " + str(mean_sw_vel_2))
    # Totels 1
    mean_totels1_1 = np.mean(first_part["totels_1"].iloc[:min(len(first_part), 150)])
    #print("totels1_1: " + str(mean_totels1_1))
    mean_totels1_2 = np.mean(last_part["totels_1"].iloc[-min(len(last_part), 150):])
    #print("totels1_2: " + str(mean_totels1_2))
    # R derivative
    mean_r_deriv_1 = np.sign(np.mean(first_part["deriv_r"]))
    #print("r_deriv_1: " + str(mean_r_deriv_1))
    mean_r_deriv_2 = np.sign(np.mean(last_part["deriv_r"]))
    #print("r_deriv_2: " + str(mean_r_deriv_2))

    param = np.array([mean_sw_vel_2 - mean_sw_vel_1, mean_totels1_2 - mean_totels1_1, mean_r_deriv_1, mean_r_deriv_2])
    variance_matrix = np.diag(PRED_VAR_DIAG)
    proba = sigmoid(np.dot(param, variance_matrix.dot(np.array(PRED_VECTOR))))
    #proba = sigmoid(np.dot(param, np.array(PRED_VECTOR)))

    print("proba outbound: " + str(proba))
    return 0 if proba > 0.7 else 2 if proba < 0.3 else 1

"""
Function used to plot several variables in order to choose biases to estimate
label

Argument:
    pandas.DataFrame() containing data for several timestamps
"""
def plot_variables(data, begin_shock, end_shock):
    # raw data
    plt.plot(data.index, data["totels_1"], 'b-')
    plt.plot(data.index, data["totels_8"], 'r-')
    plt.plot([begin_shock] * 2, [0, max(data['totels_1'])], 'g--')
    plt.plot([[end_shock]] * 2, [0, max(data['totels_1'])], 'k--')

    plt.show()

"""
Math utilitiy functions
"""
"""
Function used to detect a band of activity in which a certain data is higher
than a certain threshold with some additional feature to make detection more
stable

Input:
    List of values for the analyzed data
    Ratio threshold (detection based on the maximum value of the list)

Returns:
    Boundaries of the activity band
"""
def compute_boundary_indexes(data):
    breakpoint = compute_shock_position(data)
    if breakpoint > 37:
        return breakpoint - 37, breakpoint + 37
    return None

def compute_shock_position(data):
    c_data = data.copy().drop(['epoch', 'x', 'rho', 'deriv_r', 'SWIA_qual'], axis = 1)
    algo = rpt.Dynp(model = 'l2').fit(c_data) #min size corresponds to 2min
    breakpoints = algo.predict(n_bkps = 1)
    return breakpoints[0]

#def compute_boundary_indexes(data):
#    n = data.count()[0]
#    return int(n/2) - 37, int(n/2) + 37

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))