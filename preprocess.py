import numpy as np
import scipy
import re
from scipy.interpolate import interp1d
from scipy import signal

from einops import rearrange
def lag(X,lag_num,format):
    '''
    X: np array 
    lag_num: number of lags
    format: name of dimensions, like 'b c t f ' or 't f'
    this function is used to add lags at t dimension and merge with the f dimension
    its workflow is like this: 'b c t f -> b c t f lag -> b c t f*lag'

    Example:
    X = np.arange(24).reshape(2,3,4)
    print(X)
    X_lag = lag(X,3,'b t f')
    print(X_lag)
    '''



    #remove spaces at the beginning and end
    format = format.strip()
    #analyse format, splited by any number of spaces
    format = re.split('\s+',format)

    #find the time dimension
    time_dim = format.index('t')

    #find the feature dimension
    feature_dim = format.index('f')

    X_lags = []
    for i in range(lag_num):
        if i==0:
            X_lag = X
        else:
            #generate pad matrix
            pad_matrix_shape = list(X.shape)
            pad_matrix_shape[time_dim] = i
            pad_matrix = np.zeros(pad_matrix_shape)
            X_lag = np.concatenate((pad_matrix,X),axis=time_dim)

            #remove the last lag_num samples
            X_lag = np.delete(X_lag, np.s_[-i:], axis=time_dim)

        X_lags.append(X_lag)
    X_lags = np.concatenate(X_lags,axis=feature_dim)


    return X_lags

def align_time(array,t_origin,t_new,format):
    '''
    array: np array, the array to be aligned
    t_origin: original time points, 1d array, corresponding to the array
    t_new: new time points, 1d array
    format: name of dimensions, like 'b c t f ' or 't f'

    Example:
    X = np.arange(24).reshape(2,3,4)
    t_origin = np.arange(4)
    t_new = np.arange(0,3.9,0.1)
    X_new = align_time(X,t_origin,t_new,'b t f')
    '''


    #remove spaces at the beginning and end
    format = format.strip()
    #analyse format, splited by any number of spaces
    format = re.split('\s+',format)
    #find the time dimension
    time_dim = format.index('t')

    #compute origin frequency
    f_0 = (len(t_origin)-1)/(t_origin[-1] - t_origin[0])

    
    #compute new frequency
    f_new = (len(t_new)-1)/(t_new[-1] - t_new[0])

    # The number of samples in the resampled signal.
    sample_num = np.int32(np.round(f_new/f_0 * len(t_origin)))

    array_resample,t_resample = signal.resample(array, sample_num,t_origin,axis=time_dim)
    

    #pad the array_resample and t_resample to cover the whole range of t_new
    #find the first index of t_new that is larger than t_resample[0]
    first_index = np.where(t_new>=t_resample[0])[0][0]
    #find the last index of t_new that is smaller than t_resample[-1]
    last_index = np.where(t_new<=t_resample[-1])[0][-1]

    #pad the array_resample and t_resample
    num_pad_before = first_index
    num_pad_after = len(t_new) - last_index - 1
    if num_pad_before>0:
        pad_matrix_shape = list(array_resample.shape)
        pad_matrix_shape[time_dim] = num_pad_before
        pad_matrix = np.zeros(pad_matrix_shape)
        array_pad = np.concatenate((pad_matrix,array_resample),axis=time_dim)
        t_pad = np.concatenate((np.linspace(t_new[0],t_resample[0],num_pad_before),t_resample),axis=0)

    if num_pad_after>0:
        pad_matrix_shape = list(array_resample.shape)
        pad_matrix_shape[time_dim] = num_pad_after
        pad_matrix = np.zeros(pad_matrix_shape)
        array_pad = np.concatenate((array_resample,pad_matrix),axis=time_dim)
        t_pad = np.concatenate((t_resample,np.linspace(t_resample[-1],t_new[-1],num_pad_after)),axis=0)

    if num_pad_before<=0 and num_pad_after<=0:
        array_pad = array_resample
        t_pad = t_resample

    
    #interpolate
    interp_func = interp1d(t_pad, array_pad, axis=time_dim)
    array_new = interp_func(t_new)

    return array_new


class rearrange_and_reverse:
    def __init__(self,X,format):
        '''
        format: name of dimensions and the reduced dimensions, like 'b c t f -> (b c f) t'
        this class is used to rearrange the array to MD array
        its workflow is like this: 'b c t f -> b c t f -> b c t*f'
        Also, it can reverse the process.

        Example1:
        X = np.arange(24).reshape(1,2,3,4)
        format = 'b c t f -> (b c f) t'
        rearrane_ = rearrange_and_reverse(X,format)
        X_MD = rearrane_._ND_to_MD(X)
        X_new = rearrane_._MD_to_ND(X_MD)
        print(X_MD)
        print(X_new)

        Example2:
        X = np.arange(24).reshape(1,2,3,4)
        format = 'b c t f -> (b c f) t'
        rearrane_ = rearrange_and_reverse(X,format)
        X_MD = np.mean(X_MD,axis=-1,keepdims=True)
        X_new = rearrane_._MD_to_ND(X_MD,t=1)
        print(X_MD)
        print(X_new)
        '''
        #remove spaces at the beginning and end
        format = format.strip()

        #analyse format, splited by->
        self.format_before,self.format_after = re.split('->',format)

        #get the name and length of dimensions
        format_before = self.format_before.strip()
        format_before = re.split('\s+',format_before)
        dimnames = format_before
        dimlengths = [X.shape[format_before.index(dimname)] for dimname in format_before]

        # {dimnames:dimlengths}
        self.dimdict = dict(zip(dimnames,dimlengths))

    def _ND_to_MD(self,X):
        # X: np array
        X_MD = rearrange(X,self.format_before+'->'+self.format_after)
        return X_MD

    def _MD_to_ND(self,X_MD,**kwargs):
        # you can use this function to reverse the process of _ND_to_MD
        # X_MD: m Dimension array
        # kwargs: the dimension names and lengths of the reversed array, it will cover the original dimension names and lengths
        dimdict = self.dimdict
        dimdict.update(kwargs)
        X = rearrange(X_MD,self.format_after+'->'+self.format_before,**dimdict)
        return X





if __name__ == '__main__':
    #construct a test matrix for lag
    X = np.arange(24).reshape(2,3,4)
    print(X)
    X_lag = lag(X,3,'b t f')

    #construct a test matrix for align_time
    X = np.arange(24).reshape(2,3,4)
    t_origin = np.arange(4)
    t_new = np.arange(0,3.9,0.1)
    X_new = align_time(X,t_origin,t_new,'b t f')

    #construct a test matrix for rearrange_to_MD
    X = np.arange(24).reshape(1,2,3,4)
    format = 'b c t f -> (b c f) t'
    rearrane_ = rearrange_and_reverse(X,format)

    X_MD = rearrane_._ND_to_MD(X)
    X_new = rearrane_._MD_to_ND(X_MD)
    print(X_MD)
    print(X_new)
    X_MD = np.mean(X_MD,axis=-1,keepdims=True)
    X_new = rearrane_._MD_to_ND(X_MD,t=1)
    print(X_MD)
    print(X_new.shape)
    pass

