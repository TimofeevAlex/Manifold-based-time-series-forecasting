import numpy as np
from numpy.linalg import solve
from sklearn.metrics.pairwise import pairwise_distances

def ldmm_iteration(Y, Yt, rt, weights, h_sqr, lambd, mu, b):
    '''
    Makes a step of LDMM algorithm. Please, have a look at the paper
    for a better grasp.
    '''
    n = Yt.shape[0]
    B = np.outer(np.ones(n), weights)
    
    Wt = np.exp(-0.25/h_sqr * pairwise_distances(Yt)**2) + b * np.identity(n)
    barWt = np.multiply(B, Wt)
    Dt = np.diag(np.sum(Wt, axis=1))
    Lt = Dt - Wt
    
    Vt = Yt - rt
    U = solve(Lt + mu * barWt, mu * np.matmul(barWt, Vt))
    
    # Update Yt, rt
    Yprev = Yt
    Yt = (h_sqr * Y / lambd + mu * (U + rt)) / (h_sqr / lambd + mu)
    rt = rt + U - Yt

    return Yt, rt, Yprev

def LDMM(Y, lambd, mu, h_sqr=0.1, weights=None, n_iterations=-1, eps=1e-2, b=2):
    '''
    Performs manifold estimation by LDMM
    
    Parameters
    ----------
        Y : array-like
            Point cloud
        weights : array-like
            A 2D array of weights of the points
        h_sqr : float
            A kernel bandwidth
        lambd : float
            A penalization parameter
        mu : float
            A parameter in the Bregman iteration (it is better to take mu ~ n_instances * h)
        n_iterations : int
            A number of iterations
        eps : float
            An accuracy, matters only if n_iterations=-1, in this case the procedure 
            repeats iterations until the difference will be less than eps

    Returns
    -------
        Yt : array-like
            A recovered projection of the observed points onto the manifold
    '''
    
    n = Y.shape[0]
    D = Y.shape[1]
    # Initialize weights
    if (weights is None):
        weights = np.ones(n)
    
    # Initial values
    Yt = Y
    rt = np.zeros((n, D))
    # value on the previous step
    Yprev = np.zeros((n, D))

    # LDMM cycle
    if (n_iterations == -1):
        while (np.linalg.norm(Yt - Yprev) > eps):
            Yt, rt, Yprev = ldmm_iteration(Y, Yt, rt, weights, h_sqr, lambd, mu, b)
    else:
        for t in range(n_iterations):
            Yt, rt, Yprev = ldmm_iteration(Y, Yt, rt, weights, h_sqr, lambd, mu, b)
            

    return Yt