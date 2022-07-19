import numpy as np


def generate_modulars(num_regions, num_modules, num_regions_per_modules=None, factors=None, sigma=0.001):
    """Function for generation of matrix with different module structure
    """
    if num_regions_per_modules==None:
        num_equal = int(round(num_regions / num_modules))
        num_regions_per_modules = (num_modules - 1) * [num_equal] + [num_regions - (num_modules - 1) * num_equal]
    module_borders = [0] + list(np.cumsum(num_regions_per_modules))
    if factors is None:
        factors = np.array([[0.8,0.1,0.1],[0.1,0.8,0.1], [0.1,0.1,0.8]])
    weight_matrix = 1 + np.random.normal(0, sigma, size= (num_regions,num_regions))
    for row in range(num_modules):
        for col in range(num_modules):
            weight_matrix[module_borders[row]:module_borders[row + 1],
                module_borders[col]:module_borders[col + 1]] *= factors[row,col]
    return weight_matrix



def normalize(weight_matrix, norm_type = 'sum' ):
    
    norm_weight_matrix = weight_matrix.copy()

    if norm_type == 'cols':
        norm_weight_matrix = norm_weight_matrix / np.sum(norm_weight_matrix, axis=1)[:, None]
    elif norm_type == 'max':
        norm_weight_matrix = norm_weight_matrix / np.max( weight_matrix)
    elif norm_type == 'sum':
        norm_weight_matrix = norm_weight_matrix / np.sum( weight_matrix)

  
    return norm_weight_matrix
