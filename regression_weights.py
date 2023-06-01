import numpy as np
from sklearn import linear_model
from sklearn.cross_decomposition import PLSRegression
from ridge_via_svd import ridge_via_svd
def regress_weights(y, U, s, V, mF, normF, method, K, demean_feats):
    """
    A function that computes regression weights based on a method
    """
    assert not np.isnan(y).any()

    # de-mean data
    ym = y - np.mean(y,axis=0) if demean_feats else y

    # weights using all of the data
    if method == 'least-squares':
        B = V @ (np.diag(1./s) @ (U.T @ ym))
        
    elif method == 'pcreg': # principal components regression
        n_K = len(K)
        B = np.empty((V.shape[0], n_K))
        for j in range(n_K):
            B[:, j] = V[:, :K[j]] @ (1./s[:K[j]] * (U[:, :K[j]].T @ ym))
        
    elif method == 'ridge':
        B = ridge_via_svd(ym, U, s, V, K)
        
    elif method == 'pls':
        n_K = len(K)
        B = np.empty((U.shape[1]+1, n_K))
        for j in range(n_K):
            Z = U * s
            pls = PLSRegression(n_components=K[j])
            pls.fit(Z, ym)
            B[:, j] = pls.coef_
        B = B[1:, :]
        B = V @ B
        
    elif method == 'lasso':
        lasso = linear_model.Lasso(alpha=K, normalize=False)
        B = lasso.fit(U @ np.diag(s) @ V.T, ym).coef_
        
    else:
        raise ValueError('No valid method for {}'.format(method))

    # rescale weights to remove effect of normalization
    B = B / normF[:, np.newaxis]

    # add ones regressor
    if demean_feats:
        B = np.vstack((np.mean(y) - mF @ B, B))
    else:
        B = np.vstack((np.zeros((1, B.shape[1])), B))

    return B

if __name__ == '__main__':
    from sklearn.preprocessing import StandardScaler

    # Assume we have data y and X
    np.random.seed(0)

    X = np.random.rand(100, 10)

    true_weights = np.random.rand(10, 2)
    y = X @ true_weights + np.random.randn(100, 2)/10

    # Apply Standard Scaler to the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform SVD on scaled data
    U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)

    # Some parameters
    demean_feats = True
    method = 'least-squares'
    K = [1]  # Not used in 'least-squares', just for the function signature

    # Call function
    B = regress_weights(y, U, s, Vt.T, scaler.mean_, scaler.scale_, method, K, demean_feats)

    # Print weights
    print(B)

