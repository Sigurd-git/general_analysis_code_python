import numpy as np

def ridge_via_svd(y, U, s, V, k):
    """
    Efficiently calculates beta weights from ridge regression analysis given the
    SVD (F = U*diag(S)*V') of a feature matrix.
    """
    D = V.shape[0]
    nK = len(k)

    # correlation of principal components with the demeaned data vector
    Uty = U.T @ y

    # betas
    b = np.empty((D, nK))
    for i in range(nK):
        r = s / (s**2 + k[i])
        b[:, i] = V @ (r * Uty)  # weight principal components and transform back
    
    return b

if __name__ == '__main__':
    # Define y, U, s, V and k
    y = np.array([1, 2, 3])
    U = np.array([[0.14, 0.59, 0.80], [0.41, 0.57, -0.71], [0.89, -0.56, 0.0]])
    s = np.array([3.34, 0.51, 0.0])
    V = np.array([[0.14, 0.59, 0.80], [0.41, 0.57, -0.71], [0.89, -0.56, 0.0]])
    k = np.array([0.1, 0.5, 1.0])

    # Compute ridge regression weights
    b = ridge_via_svd(y, U, s, V, k)

    print("Ridge regression weights:")
    print(b)
