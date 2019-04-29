# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:11:58 2019

@author: natan
"""

import h5py

FILE_PATH = "../Data/datasets/boundariesALL_pFAMay2015_Dec2017.hdf5"


def translate_hdf5_epochs(filename = FILE_PATH, target_file_name = 'epoch_list'):
    f = h5py.File(filename, 'r')
    epochs = []

    # Get the data
    data = list(f['UT'])
    for i in range(len(data)):
        str_begin = str(data[i][2])[2:21]
        str_middle = str(data[i][1])[2:21]
        str_end = str(data[i][0])[2:21]
        epochs.append([str_begin, str_middle, str_end])

    tf = open(target_file_name + '.txt', 'w')
    for i in range(len(epochs)):
        tf.write(epochs[i][0] + ' ' + epochs[i][1] + ' ' + epochs[i][2] + '\n')
    tf.close()
    return epochs, data

if __name__ == '__main__':
    epochs, data = translate_hdf5_epochs()