# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 10:18:13 2019

This file containts scripts
@author: natan
"""

import pandas as pd
import random as rd
import matplotlib.pyplot as plt
import urllib
import math

import MAVEN_postprocessing as pop
import MAVEN_evaluation as ev
import MAVEN_communication_AMDA as acom
import MAVEN_prediction as pred
import TrainingManagment as tm

SAVE_PATH = "d:/natan/Documents/IRAP/Data/datasets/"
SHOCK_LIST_PATH = '../Data/datasets/ShockMAVEN_dt1h_list.txt'
#SHOCK_LIST_PATH = '../Data/datasets/ShockMAVEN_list.txt'
#SHOCK_LIST_PATH = "d:/natan/Documents/IRAP/Data/datasets/VEX_shocks/shocktime_2008.dat"

DATASET_PATH = '../Data/datasets/simple_dataset.txt'

AUX_PARAMETER_NAMES = ["mav_bkp_tot", "mav_sweakp_n", "ws_16", "mars_sw_pdyn"]
AUX_PARAMETER_COLUMNS = [["epoch", "mag"], ["epoch", "density"], ["epoch", "f107"], ["epoch", "pdyn"]]

######################################################################################################
"""
Computes the epochs of every predicted shock using the latest trained model
Inuts:
    begin_pred: begin date
    end_pred: end date
    name: Name for the output files
    manager_path: Path of the used manager
Returns:
    pred_shocks: epochs and additionnal data for each detected shock
    pred_labels:
"""
def predict_period(begin_pred, end_pred, name="../Data/amda_data/default", manager_path="../Data/managers/MAVEN_V4_opti_3"):
    pred_shocks = pd.DataFrame()
    pred_labels = pd.DataFrame()
    begins, ends = [], []

    print("Loading manager")
    manager = tm.loadManager(manager_path)

    # Subdivide in 3 days long bits
    div = (pd.Timestamp(end_pred) - pd.Timestamp(begin_pred)) / pd.Timedelta(days=3)

    begin, end = pd.Timestamp(begin_pred), pd.Timestamp(begin_pred) + pd.Timedelta(days=3)
    if end > pd.Timestamp(end_pred):
        end = pd.Timestamp(end_pred)

    for i in range(math.ceil(div)):
        print("Downloading data", str(i + 1) + '/' + str(math.ceil(div)))
        print(str(begin), "=>", str(end))
        data = acom.download_multiparam_df(str(begin).replace(' ', 'T'), str(end).replace(' ', 'T'))
        # Check if data is empty
        if data is None:
            begin, end = end, end + pd.Timedelta(days=3)
            if end > pd.Timestamp(end_pred):
                end = pd.Timestamp(end_pred)
            continue

        # Download aux data for the catalog
        aux_data = acom.download_multiparam_df(str(begin).replace(' ', 'T'), str(end).replace(' ', 'T'), param_list=AUX_PARAMETER_NAMES, param_col_names=AUX_PARAMETER_COLUMNS)

        print("Predicting")
        pred, labels = corrected_shock_prediction(manager, data)

        # Save labels
        pred_labels = pd.concat([pred_labels, labels], ignore_index = True)

        # Save interesting data
        for i in range(len(pred)):
            sample_b = next(data.loc[k] for k in range(len(data)) if abs(data.at[k, "epoch"] - pred.at[i, "begin"]) <= 2)
            aux_sample_b = next(aux_data.loc[k] for k in range(len(aux_data)) if abs(aux_data.at[k, "epoch"] - pred.at[i, "begin"]) <= 2)
            sample_e = next(data.loc[k] for k in range(len(data)) if abs(data.at[k, "epoch"] - pred.at[i, "end"]) <= 2)
            aux_sample_e = next(aux_data.loc[k] for k in range(len(aux_data)) if abs(aux_data.at[k, "epoch"] - pred.at[i, "begin"]) <= 2)

            new_row = subsample_gather_data(sample_b, aux_sample_b, sample_e, aux_sample_e)
            pred_shocks = pd.concat([pred_shocks, new_row], ignore_index=True)

        begins += [pd.Timestamp(pred.at[i, "begin"], unit='s') for i in range(len(pred))]
        ends += [pd.Timestamp(pred.at[i, "end"], unit='s') for i in range(len(pred))]
        begin, end = end, end + pd.Timedelta(days=3)
        if end > pd.Timestamp(end_pred):
            end = pd.Timestamp(end_pred)

    # Save in df
    begins, ends = [str(pd.to_datetime(begins[i], unit='s')).replace(' ', 'T') for i in range(len(begins))], [str(pd.to_datetime(ends[i], unit='s')).replace(' ', 'T') for i in range(len(ends))]
    pred_shocks["begin"], pred_shocks["end"] = begins, ends
    acom.save_df(pred_shocks, "", name + "_shocks")
    acom.save_df(pred_labels, "", name + "_labels")

    return pred_shocks, pred_labels

######################################################################################################

"""
Creates a dataset from AMDA data using preprocessing and sample labeling
Inputs:
    shock_list_path: Path of the catalog
    shock_nb: Number of shocks to predict (-1 corresponds to all of the catalog)
    random: Do you want to take randomly selected samples in the catalog ?
    offset_index: In case you set random to false, shifts the beginning index
    name: Name of the saved file
"""
def create_dataset(shock_list_path = SHOCK_LIST_PATH, shock_nb = -1, random = False, offset_index = 0, name = "default"):
    shock_list = pd.read_csv(shock_list_path)
    if shock_nb == -1:
        shock_nb = len(shock_list)
    dataset = pd.DataFrame()
    if random:
        shock_list_indexes = [k for k in range(shock_list.count()[0])]
        for i in range(shock_nb):
            random_index = shock_list_indexes[rd.randint(0, len(shock_list_indexes) - 1)]
            shock_epoch = shock_list.at[random_index, 'epoch']
            print('Loading sample ' + str(i + 1) + '/' + str(shock_nb))
            print("epoch: " + str(shock_epoch))
            try:
                shock_data = gather_and_predict_data(shock_epoch)
            except:
                print("HTTP error, trying again")
                shock_data = gather_and_predict_data(shock_epoch)
            dataset = pd.concat([dataset, shock_data], ignore_index = True)
            shock_list_indexes.remove(random_index)
            final_dataset = acom.save_df(dataset, SAVE_PATH, name)
    else:
        for i in range(offset_index, offset_index + shock_nb):
            shock_epoch = shock_list.at[i, 'epoch']
            print('Loading sample ' + str(i - offset_index + 1) + '/' + str(shock_nb))
            print("epoch: " + str(shock_epoch))
            try:
                shock_data = gather_and_predict_data(shock_epoch)
            except urllib.error.HTTPError:
                print("HTTP error, trying again")
                shock_data = gather_and_predict_data(shock_epoch)
            dataset = pd.concat([dataset, shock_data], ignore_index = True)
            final_dataset = acom.save_df(dataset, SAVE_PATH, name)
    return final_dataset
"""
Script used to train a mlp neural network easily
Inputs:
    pandas.DataFrame() data for training
    List               layers number of perceptrons
    List               layers activation function (ex: relu, tanh, softmax...)
    int                total number of epochs
    int                batch size
    str                mlp name
Returns:
    history: Training history
"""
def train_nn(manager, dataset, verbose=1):
    history = manager.run_training(dataset.drop("epoch", axis=1), verbose=verbose)
    return history

"""
Computes predictions using ANN and postprocessing
Inputs:
    manager: Manager containing the model
    dataset: pd.DataFrame Data to predict
    dt_corr: Correction time constant
    plot: Set to true to plot a simple preview of the prediction
    continuous: Set to true to predict a continuous label instead of a discrete one
"""
def corrected_prediction(manager, dataset, dt_corr=180, plot=False, continuous=False):
    if 'label' in dataset.columns:
        proba = manager.get_prob(dataset.drop(["epoch", "label"], axis=1))
    else:
        proba = manager.get_prob(dataset.drop("epoch", axis=1))
    proba["epoch"] = dataset["epoch"]
    corr_pred = pop.get_corrected_pred(proba, dt_corr, continuous)

    if plot:
        # Plot data
        if 'label' in dataset.columns:
            plt.plot(dataset.index, dataset.label, 'g-')
        plt.plot(corr_pred.index, corr_pred.label, 'b-')
        if 'label' in dataset.columns:
            plt.legend(['True label', 'Predicted label'], loc = 'upper right')
        else:
            plt.legend(['Predicted label'], loc = 'upper right')
        plt.show()
    return corr_pred

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                            UTILITY FUNCTIONS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""
Function that gather and preprocess data for a single shock epoch for dataset creation
Inputs:
    centered_epoch: Shock center epoch
Returns:
    shock_data: Features at +/- 20min from center epoch
"""
def gather_and_predict_data(centered_epoch):
    shock_begin, shock_end = str(pd.Timestamp(centered_epoch) - pd.Timedelta('20m')).replace(' ', 'T'), str(pd.Timestamp(centered_epoch) + pd.Timedelta('20m')).replace(' ', 'T')
    shock_data = acom.download_multiparam_df(shock_begin, shock_end, acom.PARAMETER_NAMES, acom.PARAMETER_COLS)
    shock_data = pred.predict_file(shock_data)
    return shock_data

"""
Creates a dataframe with one line with additionnal data
Inputs:
    sample_b: pd.Series/pd.DataFrame Features corresponding to the begining of the shock
    aux_sample_b: pd.Series/pd.DataFrame Aux data corresponding to the begining of the shock
    sample_e: pd.Series/pd.DataFrame Featres corresponding to the begining of the shock
    aux_sample_e: pd.Series/pd.DataFrame Aux data corresponding to the begining of the shock
Returns:
    new_sample: pd.DataFrame gathering all the information
"""
def subsample_gather_data(sample_b, aux_sample_b, sample_e, aux_sample_e):
    new_sample = pd.DataFrame()
    new_sample["SWIA_vel_x_b"], new_sample["SWIA_vel_x_e"] = [sample_b.SWIA_vel_x], [sample_e.SWIA_vel_x]
    new_sample["temp_b"], new_sample["temp_e"] = [sample_b.temp], [sample_e.temp]
    new_sample["density_b"], new_sample["density_e"] = [aux_sample_b.density], [aux_sample_e.density]
    new_sample["mag_b"], new_sample["mag_e"] = [aux_sample_b.mag], [aux_sample_e.mag]
    new_sample["f107_b"], new_sample["f107_e"] = [aux_sample_b.f107], [aux_sample_e.f107]
    new_sample["pdyn_b"], new_sample["pdyn_e"] = [aux_sample_b.pdyn], [aux_sample_e.pdyn]
    return new_sample

"""
Detects shocks and gather information about detected shocks
Inputs:
    manager: Manager containing the model
    dataset: Data to predict
    dt_corr: Postprocessing time constant
    dt_shock: Crossings tolerance time constant
Returns:
    corr_cross: Detected crossings
    pred_labels: Predicted labels around detected crossings
"""
def corrected_shock_prediction(manager, dataset, dt_corr=180, dt_shock=20*60):
    if "label" in dataset.columns:
        dataset = dataset.drop("label", axis=1)

    pred_data = corrected_prediction(manager, dataset, dt_corr=dt_corr, plot=False, continuous=False)
    pred_var = ev.get_category(ev.get_var(pred_data))
    corr_var = ev.corrected_var(pred_var, 15)
    pred_cross = ev.crossings_from_var(corr_var)

    corr_cross = ev.corrected_crossings(pred_cross, dt_shock)

    # Compute labels at +/- 10min
    center_epochs = [0.5 * (corr_cross.at[k, "begin"] + corr_cross.at[k, "end"]) for k in range(len(corr_cross))]
    labels = subsample_class_predict(manager, dataset, center_epochs)
    return corr_cross, labels

"""
Predicts labels for each 3-day-long dataset
Inputs:
    manager: Training manager containing the model
    data: Downloaded data used for predictions
    center_epochs: Center epochs of each detected shock
Returns:
    pd.DataFrame containing labels at +/- 10min of center epochs
"""
def subsample_class_predict(manager, data, center_epochs):
    pred = pd.DataFrame()
    for i in range(len(center_epochs)):
        # Find boudary indexes
        center_index = next(k for k in range(len(data)) if data.at[k, "epoch"] > center_epochs[i])
        begin, end = center_index - 150, center_index + 150
        if begin < 0:
            begin = 0
        if end > len(data):
            end = len(data)

        # Select data
        current_data = data.loc[begin:end]
        current_data.index = [k for k in range(len(current_data))]

        # Predict labels
        labels = corrected_prediction(manager, current_data, plot=False, continuous=True)

        # Save in df
        pred = pd.concat([pred, labels], ignore_index = True)
    return pred

if __name__ == '__main__':
    print('scripts main')