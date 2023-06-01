import numpy as np

def subdivide(N, n_divisions):
    """
    Returns indices that subdivide a vector of length N into sections of
    approximately equal length (exactly equal if n_divisions exactly subdivides N).
    
    Example
    -------
    subdivide(10, 4)
    
    Created on 2016-06-25 by Sam NH
    """

    # size of each fold
    fold_sizes = np.full(n_divisions, N // n_divisions)
    fold_sizes[:N % n_divisions] += 1
    assert sum(fold_sizes) == N

    # indices for each fold
    inds = np.empty(N)
    for i in range(n_divisions):
        xi = np.arange(fold_sizes[i]) + sum(fold_sizes[:i])
        inds[xi] = i + 1  # note: "+1" is added because Python indices start at 0, but MATLAB indices start at 1

    return inds

if __name__ == '__main__':

    print(subdivide(15, 3))  # This should return array([1., 1., 1., 1., 1., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3.])

    print(subdivide(7, 2))  # This should return array([1., 1., 1., 1., 2., 2., 2.])