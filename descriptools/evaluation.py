import numpy as np
import math


def minMaxScale(mat, mn, mx, nodata):
    scaled = np.where(mat == nodata, np.nan, mat)
    scaled = np.where(np.isnan(mat), scaled, (scaled - mn) / (mx - mn))

    return scaled


def calibration(descriptor_matrix, comparison_matrix, under):
    '''
    Return the best threshold value (th) for the linear binary classification.
    It is found by iteratively changing the value.

    Parameters
    ----------
    descriptor_matrix : int or float array
        Descriptor matrix
    comparison_matrix : int
        Benchmark flood map matrix
    under : str
        Direction which the threshold classification is applied.

    Returns
    -------
    threshold : float
        Threshold value that returns the best fit index.

    '''
    correctness_index_1, fit_index_1, _ = avaliacao(
        binary_map(descriptor_matrix, 25 / 100, under), comparison_matrix)
    correctness_index_2, fit_index_2, _ = avaliacao(
        binary_map(descriptor_matrix, 50 / 100, under), comparison_matrix)
    correctness_index_3, fit_index_3, _ = avaliacao(
        binary_map(descriptor_matrix, 75 / 100, under), comparison_matrix)

    if fit_index_3 > fit_index_2:
        if fit_index_3 > fit_index_1:
            fit_index = fit_index_3
            iteration_value = 75
        else:
            iteration_value = 25
            fit_index = fit_index_1
    else:
        if fit_index_2 > fit_index_1:
            fit_index = fit_index_2
            iteration_value = 50
        else:
            iteration_value = 25
            fit_index = fit_index_1

    for i in range(iteration_value - 20, iteration_value + 30, 10):
        correctness_index_1, iteration_fit_value, _ = avaliacao(
            binary_map(descriptor_matrix, i / 100, under), comparison_matrix)
        if iteration_fit_value >= fit_index:
            fit_index = iteration_fit_value
            threshold = i

    iteration_value = threshold
    for i in range(iteration_value - 5, iteration_value + 6, 1):
        correctness_index_1, iteration_fit_value, _ = avaliacao(
            binary_map(descriptor_matrix, i / 100, under), comparison_matrix)
        if iteration_fit_value > fit_index:
            fit_index = iteration_fit_value
            threshold = i

    iteration_value = threshold * 10
    threshold = iteration_value
    for i in range(iteration_value - 10, iteration_value + 11, 1):
        correctness_index_1, iteration_fit_value, _ = avaliacao(
            binary_map(descriptor_matrix, i / 1000, under), comparison_matrix)
        if iteration_fit_value > fit_index:
            fit_index = iteration_fit_value
            threshold = i

    iteration_value = threshold * 10
    threshold = iteration_value
    for i in range(iteration_value - 10, iteration_value + 11, 1):
        correctness_index_1, iteration_fit_value, _ = avaliacao(
            binary_map(descriptor_matrix, i / 10000, under), comparison_matrix)
        if iteration_fit_value > fit_index:
            fit_index = iteration_fit_value
            threshold = i

    return threshold / 10000


def binary_map(descriptor_matrix, threshold, under):
    '''
    Generates the linear binary map matrix from the descriptor matrix and
    threshold value

    Parameters
    ----------
    descriptor_matrix : int or float
        Terrain descriptor matrix.
    threshold : float
        Limit value used to classify the descriptors cells
    under : str
        Direction which the threshold classification is applied.

    Returns
    -------
    descriptor_binary : int8 array
        Binary flood map generated from the descriptor map.
        0 = not flooded cell, 1 = flooded cell

    '''
    descriptor_matrix = np.where(descriptor_matrix == descriptor_matrix[0, 0],
                                 np.nan, descriptor_matrix)

    if under == 'under':
        descriptor_binary = np.where(
            np.isnan(descriptor_matrix), 0,
            np.where(descriptor_matrix <= threshold, 1, 0))
    else:
        descriptor_binary = np.where(
            np.isnan(descriptor_matrix), 0,
            np.where(descriptor_matrix >= threshold, 1, 0))

    return descriptor_binary


def avaliacao(descriptor_flood_map, comparison_flood_map):
    '''
    Calculates the performance indexes: Fit index and Correctness index
    Also returns the resulting linear binary classified matrix

    Parameters
    ----------
    descriptor_flood_map : int8 array
        Binary flood map generated from the terrain descriptor.
    comparison_flood_map : int8 array
        Binary flood map generated from the benchmark flood map

    Returns
    -------
    correctness_index : float
        Correctness index.
    fit_index : float
        Fit index.
    result : int8 array
        Linear binary matrix, where 0 = true negative, 1 = false positive
        2 = false negative and 3 = true positive

    '''
    comparison_flood_map[comparison_flood_map == 1] = 2
    comparison_flood_map[comparison_flood_map == -100] = 0

    result = descriptor_flood_map + comparison_flood_map
    elements, count = np.unique(result, return_counts=True)

    if not np.any(elements == 0):
        count = np.insert(count, 0, 0)
        elements = np.insert(elements, 0, 0)
    if not np.any(elements == 1):
        count = np.insert(count, 1, 0)
        elements = np.insert(elements, 1, 1)
    if not np.any(elements == 2):
        count = np.insert(count, 2, 0)
        elements = np.insert(elements, 2, 2)
    if not np.any(elements == 3):
        count = np.insert(count, 3, 0)
        elements = np.insert(elements, 3, 3)

    correctness_index = correctness(count)
    fit_index = fit(count)

    return correctness_index, fit_index, result


def correctness(count):
    '''
    Correctness index:
        [true positive]/([false negatives] + [true positives])
    Represents the percentage of flood cells that were correctly classified.

    Parameters
    ----------
    count : int array
        Array with true/false positives/negatives.

    Returns
    -------
    correctness: float
        Correctness index.

    '''
    return ((count[3]) / (count[2] + count[3]))


def fit(count):
    '''  
    Fit index:
        [true positive]/([false negatives] + [false positives] + [true positives])
    Represents the percentage of cells that were correctly classified.

    Parameters
    ----------
    count : int array
        Array with true/false positives/negatives.

    Returns
    -------
    fit: float
        Fit index.

    '''
    return ((count[3]) / (count[3] + count[2] + count[1]))
