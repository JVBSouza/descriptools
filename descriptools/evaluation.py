import numpy as np
import math


def minMaxScale(mat, mn, mx, nodata):
    scaled = np.where(mat == nodata, np.nan, mat)
    scaled = np.where(np.isnan(mat), scaled, (scaled - mn) / (mx - mn))

    return scaled


def scale(X, x_min, x_max):
    '''
    

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    x_min : TYPE
        DESCRIPTION.
    x_max : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    x_locmin = np.unique(X)
    x_locmin = x_locmin[1]
    # # https://datascience.stackexchange.com/questions/39142/normalize-matrix-in-python-numpy
    # nom = (X-X.min())*(x_max-x_min)
    # denom = X.max() - X.min()
    # denom = denom + (denom is 0)
    # https://datascience.stackexchange.com/questions/39142/normalize-matrix-in-python-numpy
    nom = (X - x_locmin) * (x_max - x_min)
    denom = X.max() - x_locmin
    denom = denom + (denom is 0)
    return x_min + nom / denom


# import random
def calibration_new(desc, comp, under):
    '''
    Return the best threshold value (th) for the linear binary classification.
    It is found by iteratively changing the value.

    Parameters
    ----------
    desc : int or float array
        Descriptor matrix
    comp : int
        Benchmark flood map matrix
    under : str
        Direction which the threshold classification is applied.

    Returns
    -------
    th : float
        Threshold value that returns the best fit index.

    '''
    c, f1, re = avaliacao(binary_map(desc, 25 / 100, under), comp)
    c2, f2, re = avaliacao(binary_map(desc, 50 / 100, under), comp)
    c3, f3, re = avaliacao(binary_map(desc, 75 / 100, under), comp)

    if f3 > f2:
        if f3 > f1:
            f = f3
            x = 75
        else:
            x = 25
            f = f1
    else:
        if f2 > f1:
            f = f2
            x = 50
        else:
            x = 25
            f = f1

    for i in range(x - 20, x + 30, 10):
        c, fx, re = avaliacao(binary_map(desc, i / 100, under), comp)
        if fx >= f:
            f = fx
            th = i

    x = th
    for i in range(x - 5, x + 6, 1):
        c, fx, re = avaliacao(binary_map(desc, i / 100, under), comp)
        if fx > f:
            f = fx
            th = i

    x = th * 10
    th = x
    for i in range(x - 10, x + 11, 1):
        c, fx, re = avaliacao(binary_map(desc, i / 1000, under), comp)
        if fx > f:
            f = fx
            th = i

    x = th * 10
    th = x
    for i in range(x - 10, x + 11, 1):
        c, fx, re = avaliacao(binary_map(desc, i / 10000, under), comp)
        if fx > f:
            f = fx
            th = i

    return th / 10000


def calibration(desc, comp, dir):
    #Retorna o melhor percentil de corte
    # print(desc)
    x1 = 100
    x2 = 0
    best_f = 0
    threshold = 0
    c1, f1, re = avaliacao(binary_map(desc, x1, dir), comp)
    c2, f2, re = avaliacao(binary_map(desc, x2, dir), comp)
    for i in range(0, 10, 1):

        x3 = math.floor((x1 + x2) / 2)
        c3, f3, re = avaliacao(binary_map(desc, x3, dir), comp)

        if f3 > best_f:
            best_f = f3
            threshold = x3

        if f2 > f1:
            f1 = f3
            x1 = x3
        else:
            f2 = f3
            x2 = x3

    # return threshold, avaliacao(binary_map(desc,threshold,dir),comp)
    return threshold


def binary_map(desc, threshold, under):
    '''
    Generates the linear binary map matrix from the descriptor matrix and
    threshold value

    Parameters
    ----------
    desc : int or float
        Terrain descriptor matrix.
    threshold : float
        Limit value used to classify the descriptors cells
    under : str
        Direction which the threshold classification is applied.

    Returns
    -------
    binary_desc : int8 array
        Binary flood map generated from the descriptor map.
        0 = not flooded cell, 1 = flooded cell

    '''
    desc = np.where(desc == desc[0, 0], np.nan, desc)

    if under == 'under':
        binary_desc = np.where(np.isnan(desc), 0,
                               np.where(desc <= threshold, 1, 0))
    else:
        binary_desc = np.where(np.isnan(desc), 0,
                               np.where(desc >= threshold, 1, 0))

    return binary_desc


def binary_map_old(desc, threshold, under):
    #normalizar a matriz?
    desc = scale(desc, -1, 1)
    desc = np.where(desc == desc[0, 0], np.nan, desc)

    if under == 'under':
        # binary_desc = np.where(desc<=np.nanpercentile(desc,threshold),1,0)
        binary_desc = np.where(desc <= -100, 0,
                               desc <= np.nanpercentile(desc, threshold), 1, 0)
    else:
        binary_desc = np.where(desc >= np.nanpercentile(desc, threshold), 1, 0)

    return binary_desc


def avaliacao(desc, comp):
    '''
    Calculates the performance indexes: Fit index and Correctness index
    Also returns the resulting linear binary classified matrix

    Parameters
    ----------
    desc : int8 array
        Binary flood map generated from the terrain descriptor.
    comp : int8 array
        Binary flood map generated from the benchmark flood map

    Returns
    -------
    c : float
        Correctness index.
    f : float
        Fit index.
    result : int8 array
        Linear binary matrix, where 0 = true negative, 1 = false positive
        2 = false negative and 3 = true positive

    '''
    comp[comp == 1] = 2
    comp[comp == -100] = 0

    result = desc + comp
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

    c = correctness(count)
    f = fit(count)

    return c, f, result


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
    c: float
        correctness index.

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
    c: float
        correctness index.

    '''
    #intersecção do medelo e observado sobre união dos dois
    #quanto menos falsos, mais perto de 1 fica
    return ((count[3]) / (count[3] + count[2] + count[1]))
