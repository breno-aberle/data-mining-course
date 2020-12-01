import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import cdist

def rbf_kernel(data, sigma=1):
    n = data.shape[0]
    kernel_matrix = np.zeros((n,n))

    for i in range(n):
        x1 = data[i]
        for j in range(n):
            x2 = data[j]
            kernel_matrix[i][j] = np.exp( (-1)*( (x1[0]-x2[0])**2 + (x1[1]-x2[1])**2 ) / (2*sigma**2) )
    
    return kernel_matrix


def validation_method(sim_matrix, clustering_labels):

    n = sim_matrix.shape[0]
    validation_score = 0
    
    for i in range(n): # iterate through the whole dataset
        sum_numerator = 0
        sum_denominator = 0
        for j in range(n):
            if (i != j): # to make sure not the same data point
                c_ij = 1 if clustering_labels[i] == clustering_labels[j] else 0
                similarity_value = sim_matrix[i][j]
                
                sum_numerator += c_ij * similarity_value
                sum_denominator += similarity_value
                
        validation_score += sum_numerator / sum_denominator
                
    validation_score = validation_score / n           
    
    return validation_score





