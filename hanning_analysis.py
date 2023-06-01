import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from general_analysis_code_python.preprocess import generate_phoneme_features
import pandas as pd
def hanning_basis(t, n_win, hanning_width, scale):
    '''
    t: time stamp vector, t = np.arange(0, dur*sr)/sr, where dur is the duration in seconds of longer TRF and sr is sampling rate of input.
    n_win: number of hanning window
    hanning_width: width of the hanning window basis functions
    scale: scale factor of the hanning window basis functions
    '''
    n_t = len(t)
    H = np.zeros((n_t, n_win))
    for i in range(n_win):
        H[:, i] = myhann(t, hanning_width, (i-1)*0.5*hanning_width, scale)
    return H

def myhann(t, T, shift, scale):
    '''
    t: time stamp vector of the whole TRF
    T: duration of the hanning window
    shift: shift of the hanning window
    scale: scale factor of the hanning window
    '''
    y = np.zeros_like(t)
    t = t/scale - shift
    xi = (t > 0) & (t < T)
    y[xi] = 0.5*(1 - np.cos(2*np.pi*t[xi]/T))
    return y

def hanning_feature(feature,scale_factor,sr,hanning_width=None,dur=None,n_win=None):
    '''
    feature: feature matrix the shape should be time by feature
    scale_factor: scale factor of the hanning window basis functions

    You can input 2 of the 3 parameters(hanning_width,dur,n_win), and the third will be calculated.
    I recommend to input hanning_width and dur.

    dur: duration in seconds of longer TRF
    hanning_width: width of the hanning window basis functions
    n_win: number of hanning window

    return: F: feature matrix which is convolved with hanning basis. shape should be (n_t, n_win*n_f)
    '''

    ## given 2 of the 3 parameters(hanning_width,dur,n_win), calculate the third
    if hanning_width is None:
        hanning_width = 2*dur/n_win
    elif dur is None:
        dur = hanning_width*n_win/2
    elif n_win is None:
        n_win = int(dur/(hanning_width/2)-1)
    else:
        assert False,'Please input 2 of the 3 parameters(hanning_width,dur,n_win)'

    n_t,n_f = feature.shape
    # time, lag, feature
    F = np.full((n_t,n_win,n_f), np.nan)

    # time stamp vector of TRF
    t = np.arange(0, dur*sr)/sr

    # convolve feature with hanning basis
    H = hanning_basis(t, n_win, hanning_width, scale_factor)
    for k in range(n_win):
        for feature_index in range(n_f):
            F[:,k, feature_index] = np.convolve(feature[:, feature_index], H[:, k], mode='same')

    F = F.reshape(F.shape[0],-1) #reshape to time by (n_win*n_f)
    return F

def plot_feature_origin_and_constructed(feature,constructed_feature,weights,save_path):
    '''
    feature: feature vector, 1d.
    constructed_feature: feature vector which is convolved with hanning basis. shape should be time by n_win by time
    weights: weights of the hanning window basis functions
    save_path: path to save the plot
    '''
    #reconstruct feature
    reconstructed_feature = constructed_feature@weights #time by 1
    #plot
    plt.figure(figsize=(10,5))
    plt.plot(feature,label='original feature')
    plt.plot(reconstructed_feature,label='reconstructed feature')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':

# #test
#     # duration in seconds of longer TRF
#     dur = 1 #500ms

#     # width of the hanning window basis functions
#     # longer widths lead to smoother TRFs
#     hanning_width = 0.2
#     n_t=10000
#     # sampling rate of t
#     # making it high so you can see the smoothly varying underlying function
#     sr = 1000
#     n_win = int(dur/(hanning_width/2)-1)
#     weights = np.random.randn(n_win)
#     # vary this scale factor
#     scale_factor = [0,0.25,0.5,0.75, 1]

#     # time stamp vector of TRF
#     t = np.arange(0, dur*sr)/sr

#     # normal density function
#     f = np.exp(-(np.arange(n_t)-5000)**2/40000)


#     # this is like the response to fast and slow speech, in your case number of timepoints will differ which is fine
#     Y = np.full((n_t, len(scale_factor)), np.nan)

#     # time, lag, fast/slow, in your case, there will be one dimension, which is feature (i.e., frequency or phoneme identity)
#     F = np.full((n_t, n_win, len(scale_factor)), np.nan)

#     for j in range(len(scale_factor)):
#         # convolve feature with hanning basis
#         H = hanning_basis(t, n_win, hanning_width, scale_factor[j])
#         for k in range(n_win):
            
#             F[:, k, j] = np.convolve(f, H[:, k], mode='same')

#         # weight to get response
#         Y[:, j] = np.dot(F[:, :, j], weights)

#     # plot the response
#     plt.figure()
#     plt.plot(Y)
#     plt.xlabel('Time (s)')
#     plt.ylabel('Response')
#     plt.legend([f'Scale {scale_factor[0]:.2f}', f'Scale {scale_factor[1]:.2f},', f'Scale {scale_factor[2]:.2f},', f'Scale {scale_factor[3]:.2f},', f'Scale {scale_factor[4]:.2f}'])
#     plt.savefig('/scratch/snormanh_lab/shared/Sigurd/hanning_response.pdf')

#     ## Get hanning basis and plot

#     scale = 1
#     H = hanning_basis(t, n_win, hanning_width, scale)
#     plt.figure()
#     plt.plot(H)
#     h = plt.plot(np.sum(H, axis=1), 'k-')
#     plt.legend(h, ['Sum'])
#     plt.savefig('/scratch/snormanh_lab/shared/Sigurd/hanning_basis1.pdf')

#     scale = 0.5
#     H = hanning_basis(t, n_win, hanning_width, scale)
#     plt.figure()
#     plt.plot(H)
#     h = plt.plot(np.sum(H, axis=1), 'k-')
#     plt.legend(h, ['Sum'])
#     plt.savefig('/scratch/snormanh_lab/shared/Sigurd/hanning_basis0.5.pdf')

#     pass

    dur = 1
    hanning_width = 0.2
    n_win = int(dur / (hanning_width / 2) - 1)
    sr = 100
    t = np.arange(0, dur * sr) / sr
    n_t = len(t)

    # plt.figure()
    # scale = 1
    # H = hanning_basis(t, n_win, hanning_width, scale)
    # plt.plot(t, H)
    # plt.plot(t, np.sum(H, axis=1), 'k-')
    # plt.legend(['Sum'])

    # plt.figure()
    # weights = np.random.randn(n_win, 1)
    # trf = H @ weights
    # plt.plot(t, (H * weights.T))
    # plt.plot(t, trf, 'k-')
    # plt.legend(['TRF'])

    weights = np.random.randn(n_win, 1)

    def hanning_features_from_rho(F, rho_assumed, sr, hanning_width,dur):

        if rho_assumed > 0:
            scale_factor = [1, 2 ** (-rho_assumed)]
        else:
            scale_factor = [2 ** (-abs(rho_assumed)), 1]
        F_hanning = []
        for j in range(len(scale_factor)):
            F_hanning.append(hanning_feature(F[j],scale_factor[j], sr, hanning_width,  dur))
        return F_hanning
    for rho in [-1,-0.5, 0,0.5, 1]:
        if rho > 0:
            scale_factor = [1, 2 ** (-rho)]
        else:
            scale_factor = [2 ** (-abs(rho)), 1]
        stim_rates = ['Slow', 'Fast']

        # plt.figure()
        # for j in range(len(scale_factor)):
        #     H = hanning_basis(t, n_win, hanning_width, scale_factor[j])
        #     trf = H @ weights
        #     plt.subplot(1, len(scale_factor), j + 1)
        #     plt.plot(t, H * weights.T)
        #     plt.plot(t, trf, 'k-')
        #     plt.xlabel('Time (s)')
        #     plt.ylim([-5, 5])
        #     plt.title(f"{stim_rates[j]}: Scale={scale_factor[j]:.2f}")
            
        # plt.savefig('/scratch/snormanh_lab/shared/Sigurd/hanning_basis.pdf')
        # The functions hanning_features_from_rho and unravel are not provided in the MATLAB code,
        # so you need to implement them before running the remaining code.

        n_freq = 30
        F = [np.random.randn(10000, n_freq), np.random.randn(10000, n_freq)]
        Flag = hanning_features_from_rho(F, rho,sr, hanning_width, dur)
        Fcat = np.concatenate(Flag)
        weights = np.random.randn(n_win, n_freq).reshape(-1)
        Y = Fcat @ weights

        for rho_assumed in (-1,-0.5, 0,0.5, 1):
            Flag = hanning_features_from_rho(F, rho_assumed,sr, hanning_width, dur)
            Fcat = np.concatenate(Flag)
            est_weights = np.linalg.pinv(Fcat) @ Y
            Yp = Fcat @ est_weights
            r = np.corrcoef(Y, Yp)[0,1]
            print(f"rho={rho_assumed:.2f},real_rho={rho:.2f}, r={r:.2f}")
        print()
        pass

    