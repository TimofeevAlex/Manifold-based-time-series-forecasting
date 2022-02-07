import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from numpy.linalg import solve

def AMD(Y, n_neighbors, lambd, n_iterations=5):
    
    # Initialization
    X = Y[:,:]
    
    U = X[:, :]
    V = X[:, :]
    
    # Number of samples
    n = X.shape[0]
    
    for k in range(n_iterations):
    
        # Compute k-NN distances
        p_dist = pairwise_distances(X)
        p_dist_sorted = np.sort(p_dist, axis=1)

        # Determine distances to the k-th nearest neighbors
        knn_dist = p_dist_sorted[:, n_neighbors]

        # Compute bandwidths
        H = np.maximum(np.outer(knn_dist, np.ones(n)), np.outer(np.ones(n), knn_dist)) 
        H_inv = np.diag(1 / knn_dist)
        
        # Compute weights
        H2 = H**2
        W = np.exp(-0.5 * p_dist**2 / H2) * (p_dist < H)
        
        # Compute D
        d = np.sum(W, axis=1)
        D = np.diag(d)
        D_inv = np.diag(1 / d)
        
        # Compute graph Laplacian
        L = D - W
        
        b = 2 * np.max(d)
        alpha = (k+2) / 2 / b
        tau = 2 / (k+2)
        
        X = tau * V + (1 - tau) * U
        
        U = solve(b * np.eye(n) + lambd * (H_inv**2 @ L), b * X - D @ X + W @ Y)
        V_new = solve(np.eye(n) + lambd * (H_inv**2) @ (D_inv @ L), V - alpha * X + alpha * (D_inv @ (W @ Y)))
        V = V_new[:,:]
        
    # Return the denoised point cloud
    return U