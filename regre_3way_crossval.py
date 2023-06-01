from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import numpy as np
from scipy import stats
from typing import Union

def regress_predictions_from_3way_crossval(F, Y, test_folds=10, train_folds=10, method='ridge', K=None, std_feats=True, groups=None, crossval_metric='unnormalized-squared-error'):
    # Dimensions of feature matrix
    n_samples, n_features = F.shape

    # Divide signal into folds
    if isinstance(test_folds, int):
        kf = KFold(n_splits=test_folds)
        test_fold_indices = list(kf.split(np.arange(n_samples)))
    else:
        _, test_fold_indices = np.unique(test_folds, return_inverse=True)

    # Calculate predictions
    n_data_vecs = Y.shape[1]
    r = np.zeros((len(np.unique(test_fold_indices)), n_data_vecs))
    Yh = np.zeros((n_samples, n_data_vecs))

    for test_fold in np.unique(test_fold_indices):
        # train and testing folds
        test_samples = (test_fold_indices == test_fold)
        
        # within training data divide into folds
        if isinstance(train_folds, int):
            kf = KFold(n_splits=train_folds)
            train_fold_indices = list(kf.split(np.arange(sum(~test_samples))))
        else:
            _, train_fold_indices = np.unique(train_folds[~test_samples], return_inverse=True)

        # concatenate training data
        F_train = F[~test_samples]
        Y_train = Y[~test_samples]

        if method == 'least-squares':
            model = LinearRegression()
        elif method == 'ridge':
            model = Ridge()
        elif method == 'lasso':
            model = Lasso()
        elif method == 'pcreg':
            model = PCA()
        elif method == 'pls':
            model = PLSRegression()

        # fit model
        model.fit(F_train, Y_train)

        # predict on test samples
        Yh[test_samples] = model.predict(F[test_samples])

        # accuracy metrics
        r[test_fold] = np.corrcoef(Yh[test_samples].T, Y[test_samples].T)[0,1]

    # Calculate MSE
    mse = np.mean((Yh-Y)**2, axis=0)
    
    return Yh, mse, r, test_fold_indices
