import numpy as np
import pandas as pd
def pivot_cell(df,matrix_column,time_stamp=None):
    '''
    There is a time by feature matrix in each cell of matrix_column of df
    '''
    matrix = df[matrix_column].values

    #group the df rowwise
    df['row'] = np.arange(len(df))
    df = df.groupby('row')

    new_df = pd.DataFrame(matrix,columns=time_stamp)
    #melt the xs
    new_df = new_df.melt(var_name='time_stamp',value_name='diff')
    return new_df
def reshape2STRF(result_df,lag_num,):
    #result_df have column coefficients
    # will add column STRF whose shape is (feature by lag)
    coefficients = result_df['coefficients'].values
    coefficients = [coefficient[1:].reshape(lag_num,-1).T for coefficient in coefficients]
    result_df['STRF'] = coefficients
    return result_df

def split_STRF(result_df,feature_nums,feature_names,group_by=None):
    #result_df have column STRF
    # will add columns feature_names_STRF whose shape is (feature_num by lag)
    STRFs = result_df['STRF'].values
    for i,feature_num in enumerate(feature_nums):
        result_df[feature_names[i]] = [STRF[:feature_num,:] for STRF in STRFs]
        STRFs = [STRF[feature_num:,:] for STRF in STRFs]

    #melt STRFs key=feature_name, value=STRF

    result_df = result_df.drop(columns=['STRF'])

    if group_by is None:
        result_df = result_df.melt(value_vars=feature_names,var_name='feature_name',value_name='STRF')
    else:
        result_df = result_df.melt(id_vars=group_by,value_vars=feature_names,var_name='feature_name',value_name='STRF')

    return result_df

def get_TRF_single(STRF):
    #STRF is now (feature,lag)
    u, s, vh = np.linalg.svd(STRF, full_matrices=False)
    feature = u[:,0]
    time = vh[0,:]

    #correct sign to positive
    if sum(feature )<0:
        feature = -feature
    if sum(time)<0:
        time = -time

    return time
def get_TRF(result_df):
    result_df[f'TRF'] = result_df[f'STRF'].apply(get_TRF_single)
    return result_df

def average_TRF_single(TRFs):
    TRF = np.mean(TRFs,axis=0)
    return TRF


def average_TRF(result_df,group_by):

    TRF_df = result_df.groupby(group_by)['TRF'].apply(np.mean).reset_index()
    
    return TRF_df
def norm_TRF_single(TRF):
    TRF = TRF/np.max(TRF)
    return TRF
def norm_TRF(result_df):
    result_df['TRF'] = result_df['TRF'].apply(norm_TRF_single)
    return result_df