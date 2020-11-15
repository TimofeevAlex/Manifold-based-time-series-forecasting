import numpy as np
from sklearn.metrics import pairwise_distances


# Y -- (n_samples x n_features)-array of noisy observations
# n_neighbors_list -- a decreasing sequence of number of neighbors to use
# tau -- threshold parameter, must be at most half of the reach of the manifold

def SAME(Y, n_neighbors_list, tau):
    
    # number of iterations
    n_iterations = n_neighbors_list.shape[0]
    # sample size
    n = Y.shape[0]
    # dimension of the ambient space
    D = Y.shape[1]
    
    # Initialization
    X = Y.copy()
    covariances = np.tile(np.eye(D), (n, 1, 1))
    
    # pairwise distances
    dist = pairwise_distances(Y)
    
    for k in range(n_iterations):
        
        # compute adjusted distances
        a_dist = np.empty((n, n))
        for i in range(n):
            Y_diff = Y - Y[i, :]   #(n, D)
            a_dist[i,:] = np.sum( (Y_diff @ covariances[i,...] )*Y_diff, axis = 1)
        
        # Compute bandwidths
        n_neighbors = n_neighbors_list[k]
        a_dist_sorted = np.sort(a_dist, axis=1)
        h = a_dist_sorted[:, n_neighbors-1]
        # compute weights
        W = np.empty((n, n))
        for i in range(n):
            W[i] = np.exp(-a_dist[i]**2/h[i]**2) * (dist[i] < tau)
        
        # adjusted Nadaraya-Watson estimate
        X = W.dot(Y) / np.tile(np.sum(W, axis=1).reshape(-1, 1), (1, D))
        
        # compute the projectors
        if k < n_iterations-1:
            for i in range(n):

                # compute weighted covariance
                X0 = X[i, :]
                x_dist = np.linalg.norm(X - X0, axis=1)
                
                Sigma = np.cov(X.transpose(), aweights=W[i])
                covariances[i] = Sigma[:,:] / np.linalg.norm(Sigma, ord=2)
                
    return X

