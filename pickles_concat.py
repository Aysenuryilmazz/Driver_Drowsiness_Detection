#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 11:56:42 2020

@author: aysenur
"""


import os
import pandas as pd
import numpy as np

directory = "./test/"
pickles = os.listdir(directory)
pickle_list_test=[]

for pickle_file in pickles:
    clean_name = os.path.splitext(pickle_file)[0]
    if clean_name.endswith("merged_df"):
        pickle = pd.read_pickle("./test/"+pickle_file)
        pickle_list_test.append(pickle)

result_test = pd.concat(pickle_list_test)
result_test.reset_index(inplace=True, drop=True)
result_test.to_pickle("test_set.pkl")
# print(result.iloc[0])

directory = "./train/"
pickles = os.listdir(directory)
pickle_list_train=[]

for pickle_file in pickles:
    clean_name = os.path.splitext(pickle_file)[0]
    if clean_name.endswith("merged_df"):
        pickle = pd.read_pickle("./train/"+pickle_file)
        pickle_list_train.append(pickle)

result_train = pd.concat(pickle_list_train)
result_train.reset_index(inplace=True, drop=True)
result_train.to_pickle("training_set.pkl")

result_concatenated = pd.concat([result_test, result_train])
result_concatenated.reset_index(inplace=True, drop=True)
# result3.loc[:,'drowsiness']=result3.loc[:,'drowsiness'].apply(lambda x: int(x))
result_concatenated.to_pickle("whole_set.pkl")

result_dropped_columns = result_concatenated.drop(columns=['blink_no', 'blink_start_frame', 'blink_end_frame'])
# result4.loc[:,'closeness']=np.where(result4.loc[:,'avg_ear'] < 0.21, 1,0)
# result4.loc[:,'closeness']=np.where(result4.loc[:,'avg_ear'] == -1, -1,result4.loc[:,'closeness'])
result_dropped_columns.to_pickle("whole_set_selected.pkl")