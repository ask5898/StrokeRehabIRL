import pandas as pd
import os
import glob
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

PATH = r'C:/Users/ROBOTIC5/Desktop/IRL/Datasets/Dataset_TUM/'
EXPERT_PATH = r'C:\Users\ROBOTIC5\Desktop\IRL\Expert'.replace("\\","/")

#for file in glob.glob(PATH+'*Healthy*.csv') :
def create_state_csv(file) :
    df = pd.read_csv(file)
    state_df = df.drop(df.columns[0],axis=1)
    state_df.to_csv('Expert/states/'+file.split('/')[-1][:-4]+'-expert_states.csv',index=False)
 
def create_action_csv(file) :
    action = list()
    df = pd.read_csv(file)
    for column in df.columns[1:] :
        action.append(np.diff(df[column]))

    action = list(map(list,zip(*action)))
    action_df = pd.DataFrame(action, columns=df.columns[1:])
    action_df.to_csv('Expert/actions/'+file.split('/')[-1][:-4]+'-expert_actions.csv',index=False)

def append_states_and_actions(path) :
    actions = list()
    for file in glob.glob(path+'/actions/*') :
        df = pd.read_csv(file)
        _col = df.columns
        actions.append(df.values)
        
    concat_act = np.concatenate(actions,axis=0)
    act_df = pd.DataFrame(concat_act,columns=_col)
    act_df.to_csv(path+'/actions/all-actions.csv',index=False)

    states = list()
    for file in glob.glob(path+'/states/*') :
        df = pd.read_csv(file)
        states.append(df.values)
        
    concat_state = np.concatenate(states,axis=0)
    state_df = pd.DataFrame(concat_state,columns=_col)
    state_df.to_csv(path+'/states/all-states.csv',index=False)

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def interpol(arr):
    y = np.transpose(arr)
    nans, x = nan_helper(y)
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    arr = np.transpose(y)
    return arr

def remove_nan_and_normalize(path, mode) :
    arr = list()
    file = pd.read_csv(os.path.join(path,mode,'all-'+mode+'.csv'))
    for i in file.values.T:
        i = interpol(i)
        arr.append([i])
        
    arr = np.concatenate(arr,axis=0)
    arr = (arr.T - np.min(arr))/(np.max(arr) - np.min(arr))
    arr = 2*arr - 1
    arr = pd.DataFrame(arr,columns=file.columns)
    arr.to_csv(os.path.join(path,mode,'all-'+mode+'_clean.csv'),index=False)

if __name__ == '__main__' :
    #for file in glob.glob(PATH+'*Healthy*.csv') :
    append_states_and_actions(EXPERT_PATH)
    remove_nan_and_normalize(EXPERT_PATH,mode='states')
    remove_nan_and_normalize(EXPERT_PATH,mode='actions')
