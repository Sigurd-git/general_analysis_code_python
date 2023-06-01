import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import datasets
import warnings


class Ridge:
    def __init__(self, alphas=(0.1, 1.0, 10.0),scoring = 'mse', normalize = False,random_state=None):
        self.alphas = alphas
        self.random_state = random_state
        self.scoring = scoring
        self.fit_intercept = True
        self.normalize=normalize
        self.coefficients_ = None
        self.X_stds_ = None
        self.fitted_ = False


    def __normalize__(self,X,set=True):
        if set:
            self.X_stds_ = X.std(axis=0,keepdims=True)
        return X / self.X_stds_

    def fit(self, X, y,alpha = 1.0,U=None,S=None,V=None,normalize=None,fit_intercept=None):

        if normalize is None:
            normalize = self.normalize
        if fit_intercept is None:
            fit_intercept = self.fit_intercept if (U is None) or (S is None) or (V is None) else False
        # only get SVD if X provided
        if (U is None) or (S is None) or (V is None):
            if fit_intercept:
                 X = np.hstack([np.ones((X.shape[0], 1)), X])

            if self.normalize:
                X = self.__normalize__(X)

            U,S,V = np.linalg.svd(X, full_matrices=False)
            self.U_, self.S_, self.V_ = U,S,V
            self.fitted_ = True
        
        # X not provided
        else:
            if fit_intercept and not self.fitted_:
                warnings.warn("You provided U,S,V, fit_intercept=True; fit_intercept is ignored. To fit intercept, provide X,y")
        self.coefficients_ = V.T@np.linalg.inv(np.diag(S**2)+alpha*np.eye(len(S)))@np.diag(S)@U.T@y

    def inner_CV(self, X, y,folds=None,stratify=None,**kwargs):
        '''Creates (or applies, if provided) folds. For each alpha value, 
        fits ridge on the training folds and evaluates on the validation folds.
        
        Returns: a dictionary with keys alpha and values the mean score for each fold
        '''

        if folds is None:
            if stratify is None:
                # default to 5-fold CV
                folds = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            else:
                folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        else:
            if isinstance(folds,int):
                if stratify is None:
                    folds = KFold(n_splits=folds, shuffle=True, random_state=self.random_state)
                else:
                    folds = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)

            # list or tuple of indices
            elif isinstance(folds,(list,tuple)):
                if stratify is None:
                    tmp = folds
                    folds = KFold(n_splits=len(folds), shuffle=False)
                    folds.split = lambda X: tmp
            elif not isinstance(folds,(KFold,StratifiedKFold)):
                raise TypeError("inner_folds must be an integer or a KFold or a StratifiedKFold object")
            else:
                assert stratify is None, f"You provided a {type(folds)} object, stratify should not be provided"
        inner_cv = {}
        if stratify is None:
            for train_index, test_index in folds.split(X):
                # can save on SVDs by calculating once for every fold, and reusing for each alpha
                X_i,y_i = X[train_index],y[train_index]
                if self.fit_intercept:
                    X_i = np.hstack([np.ones((X_i.shape[0], 1)), X_i])
                if self.normalize:
                    X_i = self.__normalize__(X_i)
                
                U,S,V = np.linalg.svd(X_i, full_matrices=False)
                for alpha in self.alphas:
                    self.fit(X = None, y = y_i,alpha=alpha,U=U,S=S,V=V,normalize=False,**kwargs)
                    if alpha in inner_cv:
                        inner_cv[alpha].append(self.score(X[test_index], y[test_index]))
                    else:
                        inner_cv[alpha] = [self.score(X[test_index], y[test_index])]
            return {alpha:np.mean(inner_cv[alpha]) for alpha in self.alphas}
        else:
            for train_index, test_index in folds.split(X,stratify):
                # can save on SVDs by calculating once for every fold, and reusing for each alpha
                X_i,y_i = X[train_index],y[train_index]
                if self.fit_intercept:
                    X_i = np.hstack([np.ones((X_i.shape[0], 1)), X_i])
                if self.normalize:
                    X_i = self.__normalize__(X_i)
                
                U,S,V = np.linalg.svd(X_i, full_matrices=False)
                for alpha in self.alphas:
                    self.fit(X = None, y = y_i,alpha=alpha,U=U,S=S,V=V,normalize=False,**kwargs)
                    if alpha in inner_cv:
                        inner_cv[alpha].append(self.score(X[test_index], y[test_index]))
                    else:
                        inner_cv[alpha] = [self.score(X[test_index], y[test_index])]
            return {alpha:np.mean(inner_cv[alpha]) for alpha in self.alphas}

    def outer_CV(self, X, y,outer_folds = None, inner_folds=None,stratify=None):
        if outer_folds is None:
            if stratify is None:
                # default to 5-fold CV
                outer_folds = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            else:
                outer_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        else:
            if isinstance(outer_folds,int):
                if stratify is None:
                    outer_folds = KFold(n_splits=outer_folds, shuffle=True, random_state=self.random_state)
                else:
                    outer_folds = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=self.random_state)
            elif not isinstance(outer_folds,(KFold,StratifiedKFold)):
                raise TypeError("outer_folds must be an integer or a KFold or a StratifiedKFold object")
            else:
                assert stratify is None, f"You provided a {type(outer_folds)} object, stratify should not be provided"

        if stratify is None:
            scores = []
            Y_hat = np.zeros(y.shape)
            for train_index, test_index in outer_folds.split(X):
                X_i,y_i = X[train_index],y[train_index]
                inner_cv = self.inner_CV(X = X_i, y = y_i,folds=inner_folds)
                best_alpha = max(inner_cv, key=inner_cv.get)
                self.fit(X = X_i, y = y_i,alpha=best_alpha)
                scores.append(self.score(X[test_index], y[test_index]))
                test_y_hat = self.predict(X[test_index])
                Y_hat[test_index] = test_y_hat
            return Y_hat, np.mean(np.array(scores))
        else:
            scores = []
            Y_hat = np.zeros(y.shape)
            for train_index, test_index in outer_folds.split(X,stratify):
                X_i,y_i,stratify_i = X[train_index],y[train_index],stratify[train_index]
                inner_cv = self.inner_CV(X = X_i, y = y_i,folds=inner_folds,stratify=stratify_i)
                best_alpha = max(inner_cv, key=inner_cv.get)
                self.fit(X = X_i, y = y_i,alpha=best_alpha)
                scores.append(self.score(X[test_index], y[test_index]))
                test_y_hat = self.predict(X[test_index])
                Y_hat[test_index] = test_y_hat
            return Y_hat, np.mean(np.array(scores))

    def predict(self,X):
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        if self.normalize:
            X = self.__normalize__(X,set=False)
        return X @ self.coefficients_

    def score(self, X, y):
        if self.scoring == 'r':
            vx = self.predict(X)
            vx = vx - np.mean(vx)
            vy = y - np.mean(y)
            return np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))

        elif self.scoring == 'r2':
            # if y is a vector, then self.predict(X) will be a matrix but not vector. The origin for loop here is wrong.
            ss_res = np.sum((self.predict(X) - y)**2)
            ss_tot = np.sum((y - np.mean(y,axis=0,keepdims=True))**2)
            r2 = 1 - (ss_res / ss_tot)
            return r2

        elif self.scoring == 'mse':
            return np.mean((self.predict(X) - y)**2)

        # possible to pass in a custom scoring function
        elif callable(self.scoring):
            return self.scoring(self.predict(X),y)
        else:
            raise NotImplementedError("{} scoring not implemented! Create your own scoring function: def score(y,yhat):...".format(self.scoring))
    def transform(self, X):
        return self.predict(X)

    def get_params(self, deep=True): 
        return {"alphas": self.alphas, "normalize": self.normalize, "random_state": self.random_state}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


if __name__ == '__main__':
    from sklearn.linear_model import Ridge, RidgeCV
    #construct test examples for ridge
    lam_range = np.logspace(-6,6,100)
    standardize = False
    demean = False
    raw = datasets.load_diabetes()
    data = raw.data
    targets = raw.target[:,np.newaxis].reshape(-1)

    straitify = np.array([1 if i < 100 else 0 for i in range(len(targets))])

    ridge = Ridge(alphas = lam_range,normalize=standardize,scoring="r2",random_state=0)

    y_hat,r_square = ridge.outer_CV(data,targets,outer_folds=5,inner_folds=5,stratify=straitify)
    print(np.corrcoef(y_hat,targets)[0,1])
    
    #construct test examples for nested_ridge
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    Y_hat = np.zeros(targets.shape)
    for train_index, test_index in outer_cv.split(data,straitify):
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)      
        X_train, X_test,straitify_train = data[train_index], data[test_index],straitify[train_index]
        inner_cv = list(inner_cv.split(data[train_index,:],straitify[train_index]))
        y_train, y_test = targets[train_index], targets[test_index]
        ridgecv = RidgeCV(alphas = lam_range,scoring="r2",cv=inner_cv)
        ridgecv.fit(X_train,y_train)
        Y_hat[test_index] = ridgecv.predict(X_test)
    print(np.corrcoef(Y_hat,targets)[0,1])

