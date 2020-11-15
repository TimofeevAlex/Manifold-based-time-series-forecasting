import numpy as np
from preprocessing import get_dataset
from ldmm import LDMM
from same import SAME

def loc_kernel(x):
    '''
    Epanechnikov kernel         
    '''
    return 0.75 * np.maximum(1 - x**2, 0)

def shifted_weighted_kNN(timestamps_train, timestamps_test, X_train, Y_train, X_test, n_neighbors, kernel=loc_kernel, lambd=0.05):
    '''
    Makes predictions on the test dataset employing restored data by shifted weighted kNN method
    
    Parameters
    ----------
        timestamps_train : array-like
            A 1D array of timestamps for the train dataset
        timestamps_test : array-like
            A 1D array of timestamps for the test dataset
        X_train : array-like 
            A time series represented as large-dimensional vectors without the last components
        Y_train : array-like
            Answers for the train dataset
        X_test : array-like
            A time series represented as the last components of large-dimensional vectors
        n_neighbors : int, tuple
            A number of nearest neighbors for each component, if one number is given, 
            it is treated as a equal number for all components
        kernel : function 
            A function satisfying kernel properties 
        lambd : float
            A rate of discounting historical data
        
    Returns
    -------
        Y_pred : array_like
            Predictions for next values of a time series
    '''
    # increments for predicting shifted values
    n_train = Y_train.shape[0]
    A = np.diag(np.ones(n_train), 0) - np.diag(np.ones(n_train-1), -1)
    A[0,0] = 0
    increments = A.dot(Y_train)
    Y_shifted = increments + Y_train[-1]
    
    Y_pred = np.empty(Y_train.shape[1])
    dist = np.linalg.norm(X_train - X_test, axis=1)
    sorted_dist = np.sort(dist)
    
    if type(n_neighbors) == int:
        n_neighbors = [n_neighbors] * Y_train.shape[1]
        
    for i in range(Y_train.shape[1]):
        # compute weights
        h = sorted_dist[n_neighbors[i]]
        weights = kernel(dist / h)
        # reweight by the time
        weights = weights * np.exp(lambd * (timestamps_train - timestamps_test)) 
        # weighted kNN for the prediction
        Y_pred[i] = np.sum(Y_shifted[:, i] * weights) / np.sum(weights)
    
    return Y_pred

def predict_LDMM(timestamps_train, Y_train, timestamp_test, bandwidth, lambd, mu, h, n_iteration, n_neighbors, b=0):
    '''
    Makes predictions for data restored by LDMM
    
    Parameters
    ----------
        timestamps_train : array_like
            A 1D array of timestamps for the train dataset
        Y_train : array_like
            Original data
        timestamps_test : array like
            A 1D array of timestamps for the test dataset
        bandwidth : int
            The size of the bandwidth
        lambd : float
            A parameter for LDMM algorithm, take a look at LDMM function for more details
        mu : float
            A parameter for LDMM algorithm, take a look at LDMM function for more details
        h : float
            A parameter for LDMM algorithm, take a look at LDMM function for more details
        n_iterations : int
            A number of iterations for LDMM algorithm
        n_neighbors : int, tuple
            A number of nearest neighbors for each component, if one number is given, 
            it is treated as a equal number for all components
        b : float
            A parameter for LDMM algorithm, take a look at LDMM function for more details
    
    Returns
    -------
        predictions : array_like
            Predictions for future values of a time series
    '''
    # Construct the generalized features
    generalized_Y_train, generalized_Y_test = get_dataset(Y_train, bandwidth=bandwidth)
    # Construct a manifold
    generalized_Y = np.append(generalized_Y_train, generalized_Y_test.reshape(1, -1), axis=0).astype(float)
    # Normalize the numerical features
    generalized_Y = normalize(generalized_Y)
    # Find a manifold
    Z = LDMM(generalized_Y, lambd=lambd, mu=mu, h=h, n_iterations=n_iteration, b=b)  
    # Define modified train and test features
    Z_train = Z[:-1, :]
    Z_test = Z[-1, :]
    # Make predictions with shifted weighted kNN
    predictions = shifted_weighted_kNN(timestamps_train[bandwidth:], timestamp_test,\
                                       Z_train, Y_train[bandwidth:], Z_test, n_neighbors=n_neighbors)
        
    return predictions

def predict_SAME(timestamps_train, Y_train, timestamp_test, bandwidth, tau, n_iterations, n_neighbors):
    '''
    Makes predictions for data restored by SAME
    
    Parameters
    ----------
        timestamps_train : array_like
            A 1D array of timestamps for the train dataset
        Y_train : array_like
            Original data
        timestamps_test : array like
            A 1D array of timestamps for the test dataset
        bandwidth : int
            The size of the bandwidth
        tau : float
            A parameter for SAME algorithm, take a look at SAME function for more details
        n_iterations : int
            A number of iterations for SAME algorithm
        n_neighbors : int, tuple
            A number of nearest neighbors for each component, if one number is given, 
            it is treated as a equal number for all components
    
    Returns
    -------
        predictions : array_like
            Predictions for future values of a time series
    '''
    # Construct the generalized features
    generalized_X_train, generalized_X_test = get_dataset(Y_train, bandwidth=bandwidth)
    # Construct a manifold
    generalized_X = np.append(generalized_X_train, generalized_X_test.reshape(1, -1), axis=0).astype(float)
    # Normalize the numerical features
    generalized_X = normalize(generalized_X)
    Z = generalized_X
    # Find a manifold
    neighbors_list = np.array([50 * 0.93**i for i in range(n_iterations)]).astype(int)
    Z = SAME(generalized_X, neighbors_list, tau)
    # Define modified train and test features
    Z_train = Z[:-1, :]
    Z_test = Z[-1, :]
    # Make predictions with shifted weighted kNN
    predictions = shifted_weighted_kNN(timestamps_train[bandwidth:], timestamp_test,\
                                       Z_train, Y_train[bandwidth:], Z_test, n_neighbors=n_neighbors)
    return predictions

def predict_knn(timestamps_train, Y_train, timestamp_test, bandwidth, n_neighbors):
    '''
    Makes predictions without preparatory manifold restoration
    
    Parameters
    ----------
        timestamps_train : array_like
            A 1D array of timestamps for the train dataset
        Y_train : array_like
            Original data
        timestamps_test : array like
            A 1D array of timestamps for the test dataset
        bandwidth : int
            The size of the bandwidth
        n_neighbors : int, tuple
            A number of nearest neighbors for each component, if one number is given, 
            it is treated as a equal number for all components
    
    Returns
    -------
        predictions : array_like
            Predictions for future values of a time series
    '''
    # Construct the generalized features
    generalized_X_train, generalized_X_test = get_dataset(Y_train, bandwidth=bandwidth)
    # Construct a manifold
    generalized_X = np.append(generalized_X_train, generalized_X_test.reshape(1, -1), axis=0).astype(float)
    # Normalize the numerical features
    Z = normalize(generalized_X)
    # Define modified train and test features
    Z_train = Z[:-1, :]
    Z_test = Z[-1, :]
    #print(Z_train.shape, timestamps_train[bandwidth:].shape, Y_train[bandwidth:].shape)
    predictions = shifted_weighted_kNN(timestamps_train[bandwidth:], timestamp_test,\
                                       Z_train, Y_train[bandwidth:], Z_test, n_neighbors=n_neighbors)
        
    return predictions