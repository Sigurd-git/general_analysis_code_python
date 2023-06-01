import numpy as np
from scipy.linalg import svd
import sys
from print_error_message import print_error_message
def svd_for_regression(F, std_feats, demean_feats, groups=None):
    """
    Helper function for other regression scripts. Demeans and optionally z-scores the
    feature matrix, and then calculates the SVD.
    
    Created on 2016-01-10 - Made it possible to NOT demean the features and data
    """
    if groups is None:
        groups = np.arange(F.shape[1])

    assert not np.isnan(F).any()

    # optionally remove mean and standard deviation
    if demean_feats:
        mF = np.mean(F, axis=0)
    else:
        mF = np.zeros(F.shape[1])
    F_formatted = F - mF

    if std_feats:
        normF = np.std(F, axis=0)
    else:
        normF = np.ones(F.shape[1])
    normF[normF == 0] = 1
    F_formatted = F_formatted / normF

    # fix overall variance, done separately for each group
    n_groups = np.max(groups)
    if n_groups > 1:
        total_norm = np.linalg.norm(F_formatted)
        desired_group_norm = total_norm / np.sqrt(n_groups)
        for i in range(n_groups):
            xi = groups == i
            X = F_formatted[:, xi]
            group_norm = np.linalg.norm(X)
            F_formatted[:, xi] = F_formatted[:, xi] * desired_group_norm / group_norm
            normF[xi] = normF[xi] * group_norm / desired_group_norm

    try:
        U, s, Vt = svd(F_formatted, full_matrices=False)
        V = Vt.T
    except Exception as e:
        print_error_message()
        sys.exit(1)

    return U, s, V, mF, normF

if __name__ == '__main__':
    # Define a 2D feature matrix
    F = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Group array
    groups = np.array([0, 1, 2])

    # Compute the SVD with demeaning and standardizing features, and with specific groups
    U, s, V, mF, normF = svd_for_regression(F, True, True, groups)

    print("U matrix:")
    print(U)
    print("\nSingular values:")
    print(s)
    print("\nV matrix:")
    print(V)
    print("\nMean of features:")
    print(mF)
    print("\nStandard deviation of features:")
    print(normF)