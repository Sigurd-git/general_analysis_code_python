import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def plot_rasters_with_unified_scale(matrixs,plot_shape,save_path,ylabels=None,xlabels=None,titles=None):
    #matrixs: list of matrixs. Each matrix is a y by x matrix
    #plot_shape: (m,n) , which indicates that m by n subplots will be created.
    #ylabels: list of ylabels. The length of ylabels should be equal to m
    #xlabels: list of xlabels. The length of xlabels should be equal to n
    #save_path: path to save the figure

    #unify the colorbar
    vmax = -np.inf
    vmin = np.inf
    for matrix in matrixs:
        vmax = np.max([vmax,np.max(matrix)])
        vmin = np.min([vmin,np.min(matrix)])

    #plot each matrix
    fig,ax = plt.subplots(plot_shape[0],plot_shape[1],figsize=(plot_shape[1]*5,plot_shape[0]*5))

    for index,model_matrix in enumerate(matrixs):
        xindex=index%plot_shape[1]
        yindex=index//plot_shape[1]
        # assert yindex==index%plot_shape[0]
        # heatmap plot dnn_matrix
        sns.heatmap(model_matrix,ax=ax.flatten()[index],cmap='RdBu_r',cbar=False,vmin=vmin,vmax=vmax)
    
        ax.flatten()[index].set_title(titles[yindex])
        ax.flatten()[index].set_xlabel(xlabels[xindex])
        ax.flatten()[index].set_ylabel(ylabels[yindex])
        # ax.flatten()[index].set_yticks([])
        ax.flatten()[index].set_xticks([])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    #example
    import glob
    import pickle
    import os
    data_dir = '/scratch/snormanh_lab/shared/Sigurd/dnn-feats-speechAll/pickle/'
    data_files = glob.glob(data_dir+'*.pkl')
    data_files.sort()
    matrixs = []
    for data_file in data_files:
        print(data_file)
        with open(data_file,'rb') as f:
            matrix = pickle.load(f)['input_after_preproc']
            #reshape 1,1,h,w to h,w
            matrix = matrix.reshape(matrix.shape[2],matrix.shape[3])
        matrixs.append(matrix)
    
    plot_shape = (3,1)
    save_path = 'test.png'
    xlabels = ['cochleagram']
    ylabels = titles =[os.path.splitext(os.path.basename(data_file))[0] for data_file in data_files]
    plot_rasters_with_unified_scale(matrixs,plot_shape,save_path,ylabels,xlabels,titles)

