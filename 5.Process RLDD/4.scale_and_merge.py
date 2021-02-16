import pandas as pd
from sklearn.preprocessing import StandardScaler

rldd_original = pd.read_pickle('rldd_normalized.pkl')
nthu_unscaled = pd.read_pickle('final_step2.pkl')

# only scale rldd
X = rldd_original.loc[:, ["n_EAR", 
                    "n_MAR", "n_MOE", "n_EC",
                    "n_LEB", "n_SOP", "PERCLOS", "CLOSENESS"]]

y = rldd_original.loc[:, "videoname"].astype('int')
y = y.map({0: 0, 5: 0.5, 10:1, 101:1, 102:1})

# normalize each columns
scaler = StandardScaler()
scaler.fit(X)
X_scaled=scaler.transform(X)

out_rldd=pd.DataFrame(data=X_scaled, columns=["n_EAR", 
                    "n_MAR", "n_MOE", "n_EC",
                    "n_LEB", "n_SOP", "PERCLOS", "CLOSENESS"])
out_rldd.loc[:, "DROWSINESS"]=y

out_rldd.to_pickle("rldd_normalized_scaled.pkl")

# merge nthu and rldd then scale
rldd_unscaled = X.copy()
rldd_unscaled["DROWSINESS"]=y

nthu_unscaled = nthu_unscaled.loc[:, ["n_EAR", 
                    "n_MAR", "n_MOE", "n_EC",
                    "n_LEB", "n_SOP", "PERCLOS", "CLOSENESS","DROWSINESS"]]

merged_unscaled = pd.concat([rldd_unscaled,nthu_unscaled], axis=0)
merged_unscaled.reset_index(drop=True,inplace=True)

X2 = merged_unscaled.iloc[:,:-1]
y2 = merged_unscaled.iloc[:,-1]
scaler2 = StandardScaler()
scaler2.fit(X2)
X_scaled2 = scaler2.transform(X2)

out_merged=pd.DataFrame(data=X_scaled2, columns=["n_EAR", 
                    "n_MAR", "n_MOE", "n_EC",
                    "n_LEB", "n_SOP", "PERCLOS", "CLOSENESS"])
out_merged.loc[:, "DROWSINESS"]=y2

out_merged.to_pickle("merged_normalized_scaled.pkl")


