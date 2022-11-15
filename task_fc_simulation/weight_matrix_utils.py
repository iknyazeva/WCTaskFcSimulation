import numpy as np


def generate_modulars(num_regions, num_modules, num_regions_per_modules=None, factors=None, sigma=0.001):
    """Function for generation of matrix with different module structure

    Args:
        num_regions (int): number of regions generated during the simulation
        num_modules (int): number of modules
        num_regions_per_modules (list of int): number of regions in each module (should sum to num_regions)
        factors (list of list or np.ndarray): coefficient to multiply each factor

    Returns:
        weight_matrix(np.ndarray of float): resulted weight matrix
    """
    if num_regions_per_modules == None:
        num_equal = int(round(num_regions / num_modules))
        num_regions_per_modules = (num_modules - 1) * [num_equal] + [num_regions - (num_modules - 1) * num_equal]
    module_borders = [0] + list(np.cumsum(num_regions_per_modules))
    if factors is None:
        factors = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    assert np.array(factors).shape[0] == num_modules, "Number of modules should be compatible with the factos"
    weight_matrix = 1 + np.random.normal(0, sigma, size=(num_regions, num_regions))
    for row in range(num_modules):
        for col in range(num_modules):
            weight_matrix[module_borders[row]:module_borders[row + 1],
            module_borders[col]:module_borders[col + 1]] *= factors[row, col]
    return weight_matrix


def normalize(weight_matrix, norm_type='sum'):
    norm_weight_matrix = weight_matrix.copy()
    

    if norm_type == 'cols':
        norm_weight_matrix = norm_weight_matrix / np.sum(norm_weight_matrix, axis=1)[:, None]
    elif norm_type == 'max':
        norm_weight_matrix = norm_weight_matrix / np.max(weight_matrix)
    elif norm_type == 'sum':
        norm_weight_matrix = norm_weight_matrix / np.sum(weight_matrix)
    elif norm_type == 'raw':
        norm_weight_matrix  = norm_weight_matrix

    return norm_weight_matrix
