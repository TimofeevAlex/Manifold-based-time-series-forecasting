import numpy as np
from numpy.linalg import solve
from sklearn.metrics.pairwise import pairwise_distances

# Auxiliary function
#
# One iteration in the LDMM algorithm
#
# Description of the parameters is given the function LDMM below

def ldmm_iteration(X, Xt, rt, weights, h, lambd, mu, b):
    
    n = Xt.shape[0]
    B = np.outer(np.ones(n), weights)
    
    Wt = np.exp(-0.25/h * pairwise_distances(Xt)**2) + b * np.identity(n)
    barWt = np.multiply(B, Wt)
    Dt = np.diag(np.sum(Wt, axis=1))
    Lt = Dt - Wt
    
    Vt = Xt - rt
    U = solve(Lt + mu * barWt, mu * np.matmul(barWt, Vt))
    
    # Update Xt, rt
    Xprev = Xt
#     print(X)
#     print(lambd * X)
#     print(mu * (U + rt))
#     print((lambd * X + mu * (U + rt)))
#     print((lambd + mu))
    Xt = (lambd * X + mu * (U + rt)) / (lambd + mu)
#     print(Xt)
    rt = rt + U - Xt

    return Xt, rt, Xprev



# Low dimensional manifold model
#
# X -- (n_instances x n_features) point cloud
#
# weights -- (n_instances)-array of weights of the points
#
# h -- bandwidth
#
# lambd -- penalization parameter
#
# mu -- parameter in the Bregman iteration (it is better to take mu ~ n_instances * h)
#
# n_iterations -- number of iterations;
#
# eps -- accuracy; matters only if n_iterations=-1;
# in this case the procedure repeats iterations until the difference will be less than eps
#
# RETURNS: X_recovered (n_instances x n_features) -- recovered projection
# of the observed points onto the manifold

def LDMM(X, lambd, mu, h=0.1, weights=None, n_iterations=-1, eps=1e-2, b=2):
    
    n = X.shape[0]
    D = X.shape[1]
    # Initialize weights
    if (weights is None):
        weights = np.ones(n)
    
    # Initial values
    Xt = X
    rt = np.zeros((n, D))
    # value on the previous step
    Xprev = np.zeros((n, D))

    # LDMM cycle
    if (n_iterations == -1):
        while (np.linalg.norm(Xt - Xprev) > eps):
            Xt, rt, Xprev = ldmm_iteration(X, Xt, rt, weights, h, lambd, mu, b)
    else:
        for t in range(n_iterations):
            Xt, rt, Xprev = ldmm_iteration(X, Xt, rt, weights, h, lambd, mu, b)
            

    return Xt