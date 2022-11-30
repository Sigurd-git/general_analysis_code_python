from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np

def nested_crossvalidate_ridge(lst):
    '''
    X: 3D array, shape: (batch,xxx,feature)
    Y: 2D array, shape: (batch,xxx) or 3D array, shape: (batch,xxx,1)
    lam_range: lambda range
    demean: whether to demean before regression
    standardize: whether to standardize before regression
    fold_mask: the fold number assigned to each sample, which has the same length as X.shape[0]

    return: the Y_hat

    Example:
    X = np.random.rand(100,10,5)
    w = np.random.rand(5,1)
    Y = np.matmul(X,w)+np.random.rand(100,10,1)/100
    lam_range = [-10,10]
    demean = True
    standardize = True
    fold_mask = np.random.randint(0,5,100)
    lst = [lam_range, demean, standardize, fold_mask, X, Y]
    Y_hat = nested_crossvalidate_ridge(lst)
    coef = np.corrcoef(Y_hat.reshape(-1), Y.reshape(-1))[0,1]
    print(coef)
    '''
    lam_range, demean, standardize, fold_mask, X, Y = lst
    #Y 3D to 2D
    if len(Y.shape)==3:
        Y = Y.reshape(Y.shape[0], Y.shape[1])
    Y_hat = np.zeros(Y.shape)
    for testfold in np.unique(fold_mask):
        #split data
        train_val_index = np.where(fold_mask!=testfold)[0]
        test_index = np.where(fold_mask==testfold)[0]
        train_val_X = X[train_val_index]
        train_val_Y = Y[train_val_index]
        restfolds = np.unique(fold_mask)[np.unique(fold_mask)!=testfold]
        restfold_mask = fold_mask[train_val_index]
        opt_model = inner_crossvalidate_Ridge(lam_range, demean, standardize, restfolds,restfold_mask, train_val_X, train_val_Y)
        test_y_hat = opt_model.predict(X[test_index])
        Y_hat[test_index] = test_y_hat
    return Y_hat




class StandardScaler3D(StandardScaler):
    #wrap of StandardScaler to support 3d data
    def fit(self, X, y):
        if len(X.shape)==3:
            b,t,f = X.shape
            X = X.reshape(b*t, f)
            y = y.reshape(b*t, -1)
        elif len(X.shape)==2:
            pass

        return super().fit(X, y)
    def transform(self, X):
        if len(X.shape)==3:
            b,t,f = X.shape
            X = X.reshape(b*t, f)
            return super().transform(X).reshape(b,t,f)
        elif len(X.shape)==2:
            return super().transform(X)
    
class Ridge3D(Ridge):
    #wrap of Ridge to support 3d data
    def fit(self, X, y, sample_weight=None):

        if len(X.shape)==3:
            b,t,f = X.shape
            X = X.reshape(b*t, f)
            y = y.reshape(b*t)
        elif len(X.shape)==2:
            pass

        return super().fit(X, y, sample_weight)
    def predict(self, X):
        b,t,f = X.shape
        X = X.reshape(b*t, f)

        return super().predict(X).reshape(b,t)


def inner_crossvalidate_Ridge(lam_range, demean, standardize, restfolds,restfold_mask, train_val_X, train_val_Y):
    # val fold
    inner_cv = []
    for valfold in restfolds:
        val_index = np.where(restfold_mask==valfold)[0]
        train_index = np.where(restfold_mask!=valfold)[0]
        inner_cv.append((train_index,val_index))
    
    p_grid = {'ridge__alpha': np.power(2.0, np.linspace(lam_range[0], lam_range[1],20))} 
    model = Pipeline([('scaler', StandardScaler3D(with_mean=demean, with_std=standardize)), ('ridge', Ridge3D())])
    opt_model = GridSearchCV(estimator=model, param_grid=p_grid, cv=inner_cv)
    opt_model.fit(train_val_X,train_val_Y )
    return opt_model

if __name__ == '__main__':
    #construct test examples for nested_crossvalidate_ridge
    X = np.random.rand(100,10,5)
    w = np.random.rand(5,1)
    Y = np.matmul(X,w)+np.random.rand(100,10,1)/100
    lam_range = [-10,10]
    demean = True
    standardize = True
    fold_mask = np.random.randint(0,5,100)
    lst = [lam_range, demean, standardize, fold_mask, X, Y]
    Y_hat = nested_crossvalidate_ridge(lst)
    coef = np.corrcoef(Y_hat.reshape(-1), Y.reshape(-1))[0,1]
    print(coef)
