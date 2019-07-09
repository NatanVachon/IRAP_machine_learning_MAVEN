# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:55:54 2019

@author: natan
"""

import MAVEN_scripts as S

# Order is  totels_1, totels_8, deriv_r, ion_vel_x
MAVEN_STD = [73000, 42000, 980, 110]
VEX_STD = [2800, 2000, 3500, 170]

def vex_corr_pred(manager, raw_data, dt_corr = 3*60):
    data = raw_data.copy()
    data["totels_1"] *= MAVEN_STD[0] / VEX_STD[0]
    data["totels_8"] *= MAVEN_STD[1] / VEX_STD[1]
    data["deriv_r"] *= MAVEN_STD[2] / VEX_STD[2]
    data["SWIA_vel_x"] *= MAVEN_STD[3] / VEX_STD[3]

    return S.corrected_prediction(manager, data, dt_corr)