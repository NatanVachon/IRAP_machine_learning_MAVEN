# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:44:16 2019

@author: natan
"""

def save_file(data, filepath):
    file = open(filepath, 'w')
    for i in range(len(data)):
        file.write(str(data.at[i, 'epoch'] + ' '))
        file.write(str(data.at[i, 'label']) + "\n")
    file.close()
    return