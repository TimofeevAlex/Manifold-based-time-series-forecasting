import numpy as np
from sklearn.metrics import pairwise_distances

# def SAME(Y, n_neighbors_list, tau):
#     '''
#     Performs manifold estimation by SAME
    
#     Parameters
#     ----------
#         Y : array-like 
#             A 2D array of noisy observations
#         n_neighbors_list : array-like
#             A 1D decreasing sequence of number of neighbors to use
#         tau : float
#             A threshold parameter, must be at most half of the reach of the manifold
    
#     Returns
#     -------
#         X : array-like
#             A recovered projection of the observed points onto the manifold
#     '''
#     # number of iterations
#     n_iterations = n_neighbors_list.shape[0]
#     # sample size
#     n = Y.shape[0]
#     # dimension of the ambient space
#     D = Y.shape[1]
    
#     # Initialization
#     X = Y.copy()
#     covariances = np.tile(np.eye(D), (n, 1, 1))
    
#     # pairwise distances
#     dist = pairwise_distances(Y)
    
#     for k in range(n_iterations):
        
#         # compute adjusted distances
#         a_dist = np.empty((n, n))
#         for i in range(n):
#             Y_diff = Y - Y[i, :]   #(n, D)
#             a_dist[i,:] = np.sum( (Y_diff @ covariances[i,...] )*Y_diff, axis = 1)
        
#         # Compute bandwidths
#         n_neighbors = n_neighbors_list[k]
#         a_dist_sorted = np.sort(a_dist, axis=1)
#         h = a_dist_sorted[:, n_neighbors-1]
#         # compute weights
#         W = np.empty((n, n))
#         for i in range(n):
#             W[i] = np.exp(-a_dist[i]**2/h[i]**2) * (dist[i] < tau)
        
#         # adjusted Nadaraya-Watson estimate
#         X = W.dot(Y) / np.tile(np.sum(W, axis=1).reshape(-1, 1), (1, D))
        
#         # compute the projectors
#         if k < n_iterations-1:
#             for i in range(n):

#                 # compute weighted covariance
#                 X0 = X[i, :]
#                 x_dist = np.linalg.norm(X - X0, axis=1)
                
#                 Sigma = np.cov(X.transpose(), aweights=W[i])
#                 covariances[i] = Sigma[:,:] / np.linalg.norm(Sigma, ord=2)
                
#     return X


# Manifold denoising procedure
#
# Y -- (n_samples x n_features)-array of noisy observations,
# generated from the model Y_i = X_i + eps_i, 1 <= i <= n_samples,
# where X_i lies on a manifold and eps_i is a perpendicular
# zero-mean noise
#
# projectors_list -- list of initial guesses of projectors
# onto tangent spaces at the points X_1, ..., X_n
#
# bandwidths_list -- a decreasing sequence of bandwidths
#
# d -- manifold dimension
#
# tau -- threshold parameter, must be less than the reach
# of the manifold
#
def SAME(Y, projectors_list, bandwidths_list, d, tau, gamma=4.0):
    
    # number of iterations
    n_iterations = len(bandwidths_list)
    # sample size
    n = Y.shape[0]
    # dimension of the ambient space
    D = Y.shape[1]
    
    # Initialization
    X = Y[:,:]
    projectors = projectors_list
    
    for k in range(n_iterations):
        
        # pairwise distances
        dist = pairwise_distances(Y)
        
        # compute adjusted distances
        a_dist = np.empty((0, n))
        for i in range(n):
            
            # distances to the i-th point
            a_dist_i = np.linalg.norm(projectors[i] @ ((Y - Y[i,:]).transpose()), axis=0).reshape(1, -1)
            a_dist = np.append(a_dist, a_dist_i, axis=0)
            
        # compute weights
        h = bandwidths_list[k]
        W = np.exp(-a_dist**2 /h**2) * (dist < tau)
        
        # compute the adjusted Nadaraya-Watson estimate
        X = W.dot(Y) / np.tile(np.sum(W, axis=1).reshape(-1, 1), (1, Y.shape[1]))
        
        # compute the projectors
        if k < n_iterations-1:
            for i in range(n):

                # compute weighted covariance
                X0 = X[i, :]
                x_dist = np.linalg.norm(X - X0, axis=1)
                weights = 1 * (x_dist < gamma * h)
                
                X_weighted = np.diag(weights**0.5) @ (X - X0)
                u, s, vt = np.linalg.svd(X_weighted, full_matrices=False)
                
                # update the i-th projector
                projectors[i] = np.dot(vt[:d, :].transpose(), vt[:d, :])
    
    return X, projectors