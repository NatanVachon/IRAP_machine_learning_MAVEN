# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:21:59 2019

This is just a test to be familiar with AMDA requests
"""

import urllib
import io
import pandas as pd

# Username and password
USERNAME = "vachon"
PASSWORD = "virapass"

# URLs
REQUEST_URL = "http://amda.irap.omp.eu/php/rest/getParameter.php"
TOKEN_URL = "http://amda.irap.omp.eu/php/rest/auth.php"

# Sampling time
SAMPLING_TIME = 4 #s

# Saving paths
SAVE_PATH = "d:/natan/Documents/IRAP/Data/datasets/"

"""
Returns a valid token to connect to AMDA
Uses AMDA web service
"""
def get_token():
    response = urllib.request.urlopen(TOKEN_URL)
    token = response.read()
    encoding = response.headers.get_content_charset('utf-8')
    decoded_token = token.decode(encoding)
#    print(decoded_token)
    return decoded_token

"""
Builds the 'command' url to get a specific parameter between start time and end time
"""
def buildParamURL(start_time, end_time, paramID, token):
    get_url = REQUEST_URL + '?' + 'startTime='+start_time + '&stopTime='+end_time +'&parameterID='+paramID+'&token='+token+'&sampling='+str(SAMPLING_TIME)+'&userID='+USERNAME+'&password='+PASSWORD
    return get_url

"""
Very general, returns the response of a URL as a string
"""
def get_string_response(url):
#    print(url)
    response = urllib.request.urlopen(url)
    encoded = response.read()
    encoding = response.headers.get_content_charset('utf-8')
    decoded = encoded.decode(encoding)
#    print(decoded)
    return decoded

"""
Returns the URL of the file for the specified parameters
"""
def get_file_URL(start_time, end_time, paramID, token):
    param_url = buildParamURL(start_time, end_time, paramID, token)
    resp = get_string_response(param_url)
#    print(resp)
    file_url = resp.split('"')[-2]
    file_url = file_url.split('\\')
    file_url = ''.join(file_url)
#    print(file_url)
    return file_url

"""
Returns a DataFrame from a string representing an entire .txt file
"""
def get_df_from_string(file_str, column_names):
    file_str = io.StringIO(file_str)
    return pd.read_csv(file_str, comment='#', sep='\s+', names = column_names)

"""
Wraps up all the previous steps to download directly a DataFrame from user defined parameters
Uses AMDA web service
"""
def download_single_df(start_time, end_time, paramID, column_names):
    token = get_token()
    file_url = get_file_URL(start_time, end_time, paramID, token)
    file_str = get_string_response(file_url)
    df = get_df_from_string(file_str, column_names = column_names)

#    else:
#        df.columns = ['date']+[(paramID + '_'+ str(i)) for i in range(df.count(axis=1).max()-1)]
    return df

"""
Returns a dataframe of parameters in param_list between start_time and end_time
Uses AMDA web service
"""
def download_multiparam_df(start_time, end_time, param_list, param_col_names):
    dfs = []
    col_index = 0
    for i in range(len(param_list)):
        print(param_list[i] +' loading...')
        df = download_single_df(start_time, end_time, param_list[i], param_col_names[i])
        if i>0:
            df = df.iloc[:,1:]
        df_dim = df.count(axis=1)[0] + df.isna().sum(axis=1)[0]
        col_index = col_index + df_dim
        dfs.append(df)
    complete = dfs[0].join(dfs[1:])
    return complete

"""
Saves a pandas.DataFrame() to a certain path
"""
def save_df(dataset, path, name):
    saved_df = dataset.copy()
    # Convert strings to float timestamps
    for i in range(saved_df.count()[0]):
        saved_df['epoch'].at[i] = pd.Timestamp(saved_df['epoch'].at[i]).timestamp()
    # Save data
    saved_df.to_csv(path + name + '.txt', encoding = 'utf-8', index = False)
    return saved_df

"""
Main function definition
"""
# Dates
BEGIN_DATE = '2014-11-14T17:33:30'
END_DATE = '2014-11-14T19:33:30'

PARAMETER_NAMES = ["ws_0", "ws_1", "mav_bkp_mso", "ws_2", "ws_3", "mav_swiakp_vmso"]
PARAMETER_COLS = [["epoch", "rho"], ["epoch", "deriv_r"], ["epoch", "mag_x", "mag_y", "mag_z"], ["epoch", "totels_1"], ["epoch", "totels_8"], ["epoch", "SWIA_vel_x", "SWIA_vel_y", "SWIA_vel_z"]]

if __name__ == "__main__" :
    print('main AMDA')