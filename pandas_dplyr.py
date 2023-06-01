import pandas as pd
import numpy as np
def filter_at(result_df,at):
    #at is a list of dictionary, each element of this dict represent an observation
    result_dfs = []

    for observation_dict in at:
        tmp_df = result_df.copy()
        for key,value in observation_dict.items():
            tmp_df = tmp_df[tmp_df[key]==value]
        result_dfs.append(tmp_df)
    result_df = pd.concat(result_dfs)
    return result_df

def concatenate_time_by_variables(df,variables,return_foldmask=False):
### Sometimes we want to concatenate time series by variables,
### for example, we have a dataframe with columns: section, rate, rep, data
### we want to concatenate data by rate, so that the 2 regression share the same weights.

    # df is a dataframe with columns: many variables, (data or feature)
    # variables is a list of variables to combine

    #assert df is not empty
    assert not df.empty, 'df can not be empty'
    #assert variables is a list or tuple
    assert isinstance(variables,(list,tuple)), 'variables must be a list or tuple'
    column_names = df.columns
    assert 'data' in column_names or 'feature' in column_names, 'data or feature must be in the columns'
    time_variable = ('data' if 'data' in column_names else 'feature')
    tmp_df = df.copy()
    column_names = [x for x in column_names if x not in ('data','feature')]
    #make sure observations can be identified by column_names
    obs = tmp_df.groupby(column_names).apply(lambda x: len(x[time_variable])).reset_index()
    assert  (obs[0].values == 1).all(), 'observations can not be identified by rest of the columns'

    id_vars = column_names.copy()

    def concatenate_time(x,id_vars,variable):
        
        #foldmask
        #generate foldmasks
        fold_mask_list = []
        for index,maskvalue in enumerate(x[variable].values):
            fold_mask_list+= x.iloc[index][time_variable].shape[0]*[maskvalue]
        fold_mask_list = np.array(fold_mask_list)

        new_df = pd.DataFrame({'fold_mask':[fold_mask_list],time_variable:[np.concatenate(x[time_variable].values,axis=0)]},index=[0])


        return new_df
    fold_mask_dicts = []
    for variable in variables:
        
        assert variable in column_names, f'{variable} must be in the columns'
        id_vars = [x for x in id_vars if x != variable]
        #sort by variable
        tmp_df = tmp_df.sort_values(by=variable)
        tmp_df = tmp_df.groupby(id_vars).apply(lambda x: concatenate_time(x,id_vars,variable)).reset_index()
        fold_masks = tmp_df['fold_mask'].values
        column_names = [x for x in column_names if x != variable]

    #remove column level_*
    df_columns = tmp_df.columns
    df_columns = [x for x in df_columns if not x.startswith('level_')]
    #remove column fold_mask
    df_columns = [x for x in df_columns if x != 'fold_mask']
    tmp_df = tmp_df[df_columns]
    return tmp_df

def replicate_data_feature(feature_df,data_df):
#Sometimes different observations of data share the same feature,
#Like a experiment is replicated twice, or it is done by different people.
#In this case, there are more group variables in data than in feature.
#We want to replicate the feature so that the number of observations in feature and data are the same.

#Or feature have more group variables than data. This is beacure we want to test which feature can fit data better.
#We want to replicate the data so that the number of observations in feature and data are the same.

#Besides, in this way, we can make sure that the order of feature is the same for the same of data.
    #feature_df is a dataframe with columns: group variables, feature
    #data_df is a dataframe with columns: group variables, data

    #make sure observations can be identified by column_names of data
    column_names_data = data_df.columns
    column_names_feature = feature_df.columns
    assert 'data' in column_names_data, 'data must be in the columns of data'
    assert 'feature' in column_names_feature, 'feature must be in the columns of feature'
    column_names_data = [x for x in column_names_data if x != 'data']
    column_names_feature = [x for x in column_names_feature if x != 'feature']
    obs = data_df.groupby(column_names_data).apply(lambda x: len(x['data'])).reset_index()
    assert  (obs[0].values == 1).all(), 'observations can not be identified by rest of the columns of data'

    same_group_variables = [x for x in column_names_data if x in column_names_feature]

    #make sure the set of group variables of data and feature are the same
    for group_variable in same_group_variables:
        assert set(data_df[group_variable])== set(feature_df[group_variable]), f'{group_variable} of data and feature have different levels'
    
    #join data_df and feature_df
    data_feature_df = data_df.merge(feature_df,how='outer',on=same_group_variables)

    #delete column data
    feature_df = data_feature_df.drop(columns=['data'])

    #delete column feature
    data_df = data_feature_df.drop(columns=['feature'])
    
    return feature_df,data_df

if __name__=='__main__':
    # create example data
    df = pd.DataFrame({'section': ['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B','B', 'B'],
                    'rate': ['1', '1', '1', '2', '2', '2', '1', '1', '1', '2','2', '2'],
                    'rep': ['1', '2', '3', '1', '2', '3', '1', '2', '3', '1','2', '3'],
                    #range from 10*i to 10*i+10
                    'data': [np.arange(10*i,10*i+10) for i in range(12)]})
    print(concatenate_time_by_variables(df,['rate']))
    pass

