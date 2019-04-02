# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 10:18:17 2019

This file contains everything about predict labels on raw data in order to
train NNs.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt

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

TODO: VERIFIER LES INDICES

Arguments:
    pandas.DataFrame() containing data for several timestamps

Returns:
    Input dataframe with an additional column containing labels if the direction
    is detected, else return None
"""
def predict_file(data):
    # Fill nan values with median values
    data = fill_nan_values(data)
    # Compute shock boundaries
    begin_shock, end_shock = compute_boundary_indexes(data)
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
def fill_nan_values(data):
    data_copy = data.copy()
    for i in range(1, data_copy.count(1)[0]):
        data_copy.iloc[:,i] = data_copy.iloc[:,i].fillna(data_copy.iloc[:,i].median())
    return data_copy

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
    mean_sw_vel_1 = np.mean(first_part["SWIA_vel_x"])
    print('sw_vel_1 = ' + str(mean_sw_vel_1))
    mean_sw_vel_2 = np.mean(last_part["SWIA_vel_x"])
    print('sw_vel_2 = ' + str(mean_sw_vel_2))

    if mean_sw_vel_1 - mean_sw_vel_2 > SOLAR_WIND_DELTA_THRESHOLD:
        return 0
    elif mean_sw_vel_1 - mean_sw_vel_2 < SOLAR_WIND_DELTA_THRESHOLD:
        return 2
    else:
        return 1

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
    c_data = data.copy().drop(['epoch', 'r', 'lat', 'long', 'rho'], axis = 1)
    algo = rpt.Dynp(model = 'l2', min_size = 30).fit(c_data) #min size corresponds to 2min
    breakpoints = algo.predict(n_bkps = 2)
    return breakpoints[0], breakpoints[1]

def saturate(x, mini, maxi):
    return mini if x < mini else maxi if x > maxi else x