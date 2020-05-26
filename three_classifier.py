import os 
import pandas as pd
import numpy as np

# import sklearn methods 
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import sys 
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier

# display test scores and return result string and indexes of false samples
def display_test_scores(test, pred):
    str_out = ""
    str_out += ("TEST SCORES\n")
    str_out += ("\n")

    #print accuracy
    accuracy = accuracy_score(test, pred)
    str_out += ("ACCURACY: {:.4f}\n".format(accuracy))
    str_out += ("\n")

    #print AUC score
    auc = roc_auc_score(test, pred)
    str_out += ("AUC: {:.4f}\n".format(auc))
    str_out += ("\n")

    #print confusion matrix
    str_out += ("CONFUSION MATRIX:\n")
    conf_mat = confusion_matrix(test, pred)
    str_out += ("{}".format(conf_mat))
    str_out += ("\n")
    str_out += ("\n")

    #print FP, FN
    str_out += ("FALSE POSITIVES:\n")
    fp = conf_mat[1][0]
    pos_labels = conf_mat[1][0]+conf_mat[1][1]
    str_out += ("{} out of {} positive labels ({:.4f}%)\n".format(fp, pos_labels,fp/pos_labels))
    str_out += ("\n")

    str_out += ("FALSE NEGATIVES:\n")
    fn = conf_mat[0][1]
    neg_labels = conf_mat[0][1]+conf_mat[0][0]
    str_out += ("{} out of {} negative labels ({:.4f}%)\n".format(fn, neg_labels, fn/neg_labels))
    str_out += ("\n")

    #print classification report
    str_out += ("PRECISION, RECALL, F1 scores:\n")
    str_out += ("{}".format(classification_report(test, pred)))
    
    false_indexes = np.where(test != pred)
    return str_out, false_indexes


################ DECISION TREE ###################
df_original = pd.read_pickle('final_step2.pkl')

X = df_original.loc[:, ["n_EAR", 
                    "n_MAR", "n_MOE", "n_EC",
                    "n_LEB", "n_SOP", "PERCLOS", "CLOSENESS"]]

y = df_original.loc[:, "DROWSINESS"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)

# cross-validation with 10 splits
cv = StratifiedShuffleSplit(n_splits=10, random_state = 42)

# CART decision tree
cart = DecisionTreeClassifier(random_state = 0)

# parameters 
parameters = {
                "criterion": ["gini","entropy"],
                "splitter": ["best","random"],
                "class_weight": [None, "balanced"],
                }

# grid search for parameters
grid = GridSearchCV(estimator=cart, param_grid=parameters, cv=cv, n_jobs=-1)
grid.fit(X_train, y_train)

# print best scores
print("The best parameters are %s with a score of %0.4f"
      % (grid.best_params_, grid.best_score_))

# prediction results
y_pred = grid.predict(X_test)

# print accuracy metrics
results, false = display_test_scores(y_test, y_pred)
print(results)

################ NAIVE-BAYES ###################
# parameters 
parameters = {}

# grid search for parameters
grid2 = GridSearchCV(estimator=GaussianNB(), param_grid=parameters, cv=cv, n_jobs=-1)
grid2.fit(X_train, y_train)

# print best scores
print("The best parameters are %s with a score of %0.4f\n"
      % (grid2.best_params_, grid2.best_score_))

# prediction results
y_pred2 = grid2.predict(X_test)

# print accuracy metrics
results2, false2 = display_test_scores(y_test, y_pred2)
print(results2)
    
################ K-NN ###################
knn = KNeighborsClassifier(weights="distance", n_neighbors=25)
# parameters 
parameters = {
    # "weights": ["uniform", "distance"],
    # "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
    # "n_neighbors": [5,10,15,20,25,30]
    }

# grid search for parameters
grid3 = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=parameters, cv=cv, n_jobs=-1)
grid3.fit(X_train, y_train)

# print best scores
print("The best parameters are %s with a score of %0.4f\n"
      % (grid3.best_params_, grid3.best_score_))

# prediction results
y_pred3 = grid3.predict(X_test)

# print accuracy metrics
results3, false3 = display_test_scores(y_test, y_pred3)
print(results3)
