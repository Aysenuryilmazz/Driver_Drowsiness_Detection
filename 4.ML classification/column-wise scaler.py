import pandas as pd
from sklearn.preprocessing import StandardScaler

df_original = pd.read_pickle('final_step2.pkl')

X = df_original.loc[:, ["n_EAR", 
                    "n_MAR", "n_MOE", "n_EC",
                    "n_LEB", "n_SOP", "PERCLOS", "CLOSENESS"]]

y = df_original.loc[:, "DROWSINESS"]

# normalize each columns
scaler = StandardScaler()
scaler.fit(X)
X_scaled=scaler.transform(X)

out_df=pd.DataFrame(data=X_scaled, columns=["n_EAR", 
                    "n_MAR", "n_MOE", "n_EC",
                    "n_LEB", "n_SOP", "PERCLOS", "CLOSENESS"])
out_df.loc[:, "DROWSINESS"]=y

#out_df.to_pickle("final_step2_scaled.pkl")

pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', 500)
out_df.describe()

import pickle
scalerfile = 'scaler.sav'
pickle.dump(scaler, open(scalerfile, 'wb'))

scalerfile = 'scaler.sav'
scaler = pickle.load(open(scalerfile, 'rb'))
test_scaled_set = scaler.transform(X)