# -*- coding: utf-8 -*-
"""
Created on Sat May 16 22:04:01 2020

@author: hakko
"""
import pandas as pd
import numpy as np

df = pd.read_pickle('whole_set_selected2_preprocessed_normalized.pkl')

df.drop(columns=['CLOSENESS'], inplace=True)
df.loc[:,'CLOSENESS']=np.where(df['n_EAR'] < df['n_021'], 1,0)

# df.loc[:,'PERCLOS2'] = df.loc[:,'CLOSENESS'].rolling(min_periods=1, window=150).sum()/150



df.to_pickle('whole_set_selected2_preprocessed_normalized_fixed.pkl')