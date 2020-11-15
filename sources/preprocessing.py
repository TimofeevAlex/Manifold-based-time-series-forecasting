import numpy as np

def ts_to_vec(ts, bandwidth=12):
    '''
    Converts time series to large-dimensional vectors suitable 
    for manifold-based models
    
    Parameters
    ----------
        ts : array-like
            A 2D array, representing a multivariate time series
        bandwidth : int
            The size of the bandwidth
    
    Returns
    -------
        out : array-like
            An array of large-dimensional vectors
    '''
    out = ts[:, :]
    for i in range(bandwidth-1):
        out = np.append(out[:-1, :ts.shape[1]], out[1:, :], axis=1)
        
    return out

def get_dataset(ts, bandwidth=12):
    '''
    Splits a times series represented as large-dimensional vectors
    onto train and test
    
    Parameters
    ----------
        ts : array-like
            A 2D array, representing a multivariate time series
        bandwidth : int
            The size of the bandwidth
    
    Returns
    -------
        X_train : array-like
            An array of large-dimensional vectors
        X_test : array-like
            An array of large-dimensional vectors
    '''
    X = ts_to_vec(ts, bandwidth=bandwidth)
    
    X_train = X[:-1, :]
    X_test = X[-1, :].reshape(-1)
    
    return X_train, X_test