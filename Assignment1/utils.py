import numpy as np

def zscore(array):
    mean = np.mean(array)
    std = np.std(array)
    
    array_scaled = (array - mean) / std
    
    return array_scaled