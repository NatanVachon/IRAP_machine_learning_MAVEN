# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:17:56 2019

@author: natan
"""
import pandas as pd

import MAVEN_communication_AMDA as acom
import MAVEN_prediction as pred

def detect_shocks(begin_time, end_time):
    # Download shock dist
    shock_data = acom.download_multiparam_df(begin_time, end_time, ["ws_9"], [["epoch", "shock_dist"]])
    # Find crossings
    crossings = []
    passFlag = False
    for i in range(shock_data.count()[0] - 1):
        if passFlag:
            passFlag = False
            continue
        if shock_data.at[i, "shock_dist"] * shock_data.at[i + 1, "shock_dist"] <= 0.0:
            crossings.append(shock_data.at[i, "epoch"])
            passFlag = True
    print(crossings)

    shocks_epoch = []
    for i in range(len(crossings)):
        print("Sample " + str(i+1) + '/' + str(len(crossings)))
        # Download window around shock
        shock_begin, shock_end = str(pd.Timestamp(crossings[i]) - pd.Timedelta('20m')).replace(' ', 'T'), str(pd.Timestamp(crossings[i]) + pd.Timedelta('20m')).replace(' ', 'T')
        data = acom.download_multiparam_df(shock_begin, shock_end, acom.PARAMETER_NAMES, acom.PARAMETER_COLS)
        # Detect shock time
        shock_index = pred.compute_shock_position(data)
        shocks_epoch.append(data.at[shock_index, "epoch"][:19])
    return shocks_epoch, shock_data