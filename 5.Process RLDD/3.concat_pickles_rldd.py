import os
import pandas as pd
import numpy as np


df_list=[]

directory = "./"
files = os.listdir(directory)
for file in files:
    clean_name = os.path.splitext(file)[0]
    if clean_name.startswith('Fold') and clean_name.endswith('frame_info_df'):
        df = pd.read_pickle(file)
        df_list.append(df)

df_original = pd.concat(df_list)

# drop some -1s in columns due to multiple face detected
df = df_original.loc[df_original['face_detected'] == 1]

# discard first 150 frame
# df.loc[df['reserved_for_calibration']== True, 'perclos']=0
df = df[df['reserved_for_calibration']== False]

# check if are there any -1 in all columns
df.eq(-1).all()

# theres no -1 left so reset indexes
df = df.reset_index(drop=True)

# check if left and right eyes' values are differ each other
diff = df['left_ear']-df['right_ear']
np.where(np.logical_and(diff>0.21, df['avg_ear']<0.21))

z_score = (diff - diff.mean())/diff.std(ddof=0)
np.where(z_score>3)
np.where(np.logical_and(z_score>3, df['avg_ear']<0.21))

# some samples which are one eye is 0, are detected.
print(df.iloc[747781][['left_ear','right_ear','avg_ear']])

# check if is there more 0 value
a=np.where(np.logical_and(df['left_ear']==0, df['right_ear']!=0))
b=np.where(np.logical_and(df['left_ear']!=0, df['right_ear']==0))

# yes, there are more
for i in a:
    print(df.iloc[i][['left_ear','right_ear','avg_ear']])
    
for j in b:
    print(df.iloc[j][['left_ear','right_ear','avg_ear']])
    
# use not 0 column instead of avg_ear
df['EAR'] = df['avg_ear']
for i in a:
    df.loc[i,'EAR'] = df['right_ear']
for j in b:
    df.loc[i,'EAR'] = df['left_ear']
    
# do the same with other eye columns
a=np.where(np.logical_and(df['left_ec']==0, df['right_ec']!=0))
b=np.where(np.logical_and(df['left_ec']!=0, df['right_ec']==0))
df['EC'] = df['avg_ec']
for i in a:
    df.loc[i,'EC'] = df['right_ec']
for j in b:
    df.loc[i,'EC'] = df['left_ec']
    
a=np.where(np.logical_and(df['left_leb']==0, df['right_leb']!=0))
b=np.where(np.logical_and(df['left_leb']!=0, df['right_leb']==0))
df['LEB'] = df['avg_leb']
for i in a:
    df.loc[i,'LEB'] = df['right_leb']
for j in b:
    df.loc[i,'LEB'] = df['left_leb']
    
a=np.where(np.logical_and(df['left_sop']==0, df['right_sop']!=0))
b=np.where(np.logical_and(df['left_sop']!=0, df['right_sop']==0))
df['SOP'] = df['avg_sop']
for i in a:
    df.loc[i,'SOP'] = df['right_sop']
for j in b:
    df.loc[i,'SOP'] = df['left_sop']
    
# check for more anormal differences
np.where(np.logical_and(z_score>3, df['avg_ear']<0.3))

# one sample. What can we do about it? I guess it's better to use avg_ear for cases like these.
print(df.iloc[1122506][['left_ear','right_ear','avg_ear']])

# check if is there 0 in mar, moe, perclos and closeness
df['mar'].describe()
len(np.where(df['mar']==0)[0])

df['moe'].describe()
len(np.where(df['moe']==0)[0])

df['perclos'].describe()
len(np.where(df['perclos']==0)[0])

df['closeness'].describe()
len(np.where(df['closeness']==0)[0])

# MAR is acceptable. 
# closeness will be updated after normalization nothing can be done for now.
# fixing perclos should be done in process_video.py along with all the steps in preprocessing
# it's possible to fix MOE with new values of EAR
# first rename columns
df.rename(columns={"mar": "MAR", "moe": "MOE", "perclos": "PERCLOS", "closeness": "CLOSENESS", "drowsiness":"DROWSINESS"}, inplace=True)

df['MOE'] = df ['MAR'] / df['EAR']

user_list = list(df['subject'].unique())

list_of_all_original=[]
for user in user_list:
    list_of_all_original.append(df.groupby('subject').get_group(user))
    
list_of_all_first_90 = []
for user in user_list:
    list_of_all_first_90.append(df.groupby('subject').get_group(user)[:90])


##### standart scaler çalıştırma
from sklearn.preprocessing import StandardScaler

en_son_df_list=[]
for i in range(len(list_of_all_first_90)):
    scaler = StandardScaler()
    scaler.fit(list_of_all_first_90[i].loc[ : , ["left_ear", "right_ear", "avg_ear", 
                                              "left_ec", 
                                              "right_ec","avg_ec", "left_leb", "right_leb", 
                                              "avg_leb","left_sop", "right_sop", "avg_sop",
                                              "EAR","EC","LEB","SOP","MAR","MOE"]])
    son_df=pd.DataFrame(scaler.transform(list_of_all_original[i].loc[ : , ["left_ear", "right_ear", "avg_ear", 
                                              "left_ec", 
                                              "right_ec","avg_ec", "left_leb", "right_leb", 
                                              "avg_leb","left_sop", "right_sop", "avg_sop",
                                              "EAR","EC","LEB","SOP","MAR","MOE"]]))
    
    
    
    son_df.columns=["n_left_ear", "n_right_ear", "n_avg_ear", 
                    "n_left_ec", 
                    "n_right_ec","n_avg_ec", "n_left_leb", "n_right_leb", 
                    "n_avg_leb","n_left_sop", "n_right_sop", "n_avg_sop",
                    "n_EAR","n_EC","n_LEB","n_SOP","n_MAR","n_MOE"]
    
    
    son_df['subject']=user_list[i]    
    en_son_df_list.append(son_df) 

# scale 021 by using EAR
for i in range(len(list_of_all_first_90)):
    mean=list_of_all_first_90[i].loc[:,'EAR'].mean()
    std=list_of_all_first_90[i].loc[:,'EAR'].std()
    threshold = 0.21
    n_threshold = (threshold-mean)/std
    print (mean, std, threshold, n_threshold)
    en_son_df_list[i].loc[:,'n_021'] = n_threshold
    # fix closeness column with newly updated ear threshold
    en_son_df_list[i].loc[:,'CLOSENESS']=np.where(en_son_df_list[i]['n_EAR'] < en_son_df_list[i]['n_021'], 1,0)
    # fix perclos column also
    en_son_df_list[i].loc[:,'PERCLOS']=en_son_df_list[i].loc[:,'CLOSENESS'].rolling(min_periods=1, window=150).sum()/150

con_df = pd.concat(en_son_df_list)

# drop original CLOSENESS and PERCLOS then merge two dataframes
df.drop(columns=['CLOSENESS','PERCLOS'], inplace=True)
con_df.drop(columns=['subject'], inplace=True)
merged = pd.concat([df.reset_index(drop=True), con_df.reset_index(drop=True)], axis=1)

merged_selected_cols=merged[["fold","subject","videoname","frame_no","MAR","MOE","EAR","EC","LEB","SOP",
                       "n_EAR","n_EC","n_LEB","n_SOP","n_MAR","n_MOE","CLOSENESS","PERCLOS"]]


merged_selected_cols.to_pickle("rldd_normalized.pkl")