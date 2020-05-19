#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:57:52 2020

@author: aysenur
"""


import os 
import pandas as pd
import numpy as np
import sys 
import matplotlib.pyplot as plt

# import sklearn methods 
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from utils3 import *


"""Prepare Dataset"""
df_original = pd.read_pickle('one_eye_with_still_problem.pkl')

#drop rows == -1 in drowsiness
df_original = df_original.loc[ np.logical_or(df_original['drowsiness'] == 1, df_original['drowsiness'] == 0 ) ]
df_original.loc[df_original['drowsiness'] == 1, 'drowsiness'].count()

X = df_original.loc[:, ["n_avg_ear", 
                    "n_mar", "n_moe", "n_avg_eye_circularity",
                    "n_avg_leb", "n_avg_sop", "perclos"]]

y = df_original.loc[:, "drowsiness"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)

# cross-validation with 5 splits
cross_val = StratifiedShuffleSplit(n_splits=10, random_state = 42)




"""Hyperparameter Searching"""

rf = RandomForestClassifier(random_state = 0,criterion='entropy',class_weight='balanced',max_features=None)
# parameters 
parameters = {  
                "n_estimators":[20],
                #'max_depth': list(range(1,11)),
                #"criterion": ["gini","entropy"],
                #"class_weight": [None, "balanced"],
                #"max_features":["auto", None, "log2"],
                }

# grid search for parameters
grid = GridSearchCV(estimator=rf, param_grid=parameters, cv=cross_val, n_jobs=-1)#multithreading; all cores are used
grid.fit(X_train, y_train)

# print best scores
print("The best parameters are %s with a score of %0.4f"
      % (grid.best_params_, grid.best_score_))


"""Predictions"""

# prediction results
y_pred = grid.predict(X_test)

# print accuracy metrics
results, false = display_test_scores_v2(y_test, y_pred)
print(results)

"""Show feature importance"""

# pie-chart 
labels = X_train.columns
plt.pie(grid.best_estimator_.feature_importances_, labels=labels, shadow=True, startangle=90) 
