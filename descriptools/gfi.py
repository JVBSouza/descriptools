from numba import cuda, jit, float32
import numpy as np
import math

from descriptools.helpers import divisor


def geomorphic_flood_index_sequential(hand, flow_accumulation, indices,
                                      expoent, scale_factor, px):
    '''
    Sequential method for the GFI index

    Parameters
    ----------
    hand : int or flat array
        Height above the nearest drainage.
    flow_accumulation : int
        Flow accumulation. Represent the river as source of hazard.
    indices : int array
        River cell index.
    expoent : float
        Expoent (<1) Calibrated for the region.
    scale_factor : float
        Scale factor.

    Returns
    -------
    geomorphic_flood_index : float array
        geomorphic flood index array.

    '''
    river_flow_accumulation = river_accumulation(flow_accumulation, indices)

    geomorphic_flood_index = np.where(
        hand == -100, -100,
        np.where(
            hand == 0, 0,
            np.log(scale_factor * (np.power(
                np.where(river_flow_accumulation == 0, 1,
                         river_flow_accumulation) * (px * px), expoent)) / hand)))

    return geomorphic_flood_index


@jit
def geomorphic_flood_index_sequential_jit(hand, flow_accumulation, indices,
                                          expoent, scale_factor, px):
    river_flow_accumulation = river_accumulation(flow_accumulation, indices)
    geomorphic_flood_index = np.zeros(hand.shape, dtype=float32)

    for i in range(0, len(hand), 1):
        for j in range(0, len(hand[0]), 1):
            if hand[i, j] == -100:
                geomorphic_flood_index[i, j] = -100
            else:
                geomorphic_flood_index[i, j] = np.log(
                    scale_factor *
                    (np.power(river_flow_accumulation[i, j] * (px * px), expoent)) /
                    (hand[i, j] + 0.01))

    return geomorphic_flood_index


@jit
def ln_hl_H_sequential_jit(hand, flow_accumulation, expoent, scale_factor, px):

    ln_hl_H = np.zeros(hand.shape, dtype=float32)

    for i in range(0, len(hand), 1):
        for j in range(0, len(hand[0]), 1):
            if hand[i, j] == -100:
                ln_hl_H[i, j] = -100
            else:
                if flow_accumulation[i, j] == 0:
                    ln_hl_H[i,
                            j] = np.log(scale_factor * (np.power(1 * (px * px), expoent)) /
                                        (hand[i, j] + 0.01))
                else:
                    ln_hl_H[i, j] = np.log(
                        scale_factor *
                        (np.power(flow_accumulation[i, j] * (px * px), expoent)) /
                        (hand[i, j] + 0.01))

    return ln_hl_H


def ln_hl_H_sequential(hand, flow_accumulation, expoent, scale_factor, px):
    '''
    Sequential method for the ln(hl/HAND) index

    Parameters
    ----------
    hand : int or flat array
        Height above the nearest drainage.
    flow_accumulation : int
        Flow accumulation. Represent the upslope area as source of hazard.
    expoent : float
        Expoent (<1) Calibrated for the region.
    scale_factor : float
        Scale factor.

    Returns
    -------
    ln_hl_H : float array
        ln(hl/H) index array.

    '''
    ln_hl_H = np.where(
        hand == -100, -100,
        np.where(
            hand == 0, 0,
            np.log(scale_factor * (np.power(
                np.where(flow_accumulation == 0, 1, flow_accumulation) * (px * px),
                expoent)) / hand)))
    return ln_hl_H


@jit
def river_accumulation(flow_accumulation, indices):
    '''
    Method that return the array with the river cell flow accumulation.

    Parameters
    ----------
    flow_accumulation : int
        Flow accumulation. 
    indices : indices : int array
        River cell index.

    Returns
    -------
    river_flow_accumulation : int
        Flow accumulation of the river cell. Represent the river as source of hazard.

    '''
    row, col = flow_accumulation.shape
    flow_accumulation = np.asarray(flow_accumulation).reshape(-1)
    indices = np.asarray(indices).reshape(-1)
    river_flow_accumulation = np.zeros(row * col, float32)

    river_flow_accumulation = np.where(indices != -100,
                                       flow_accumulation[indices],
                                       flow_accumulation[0])

    river_flow_accumulation = river_flow_accumulation.reshape(row, col)

    return river_flow_accumulation


def gfi_calculator(hand, 
                   flow_accumulation, 
                   indices, 
                   n_gfi, 
                   scale_factor,
                   size,
                   division_column=0, 
                   division_row=0):
    '''
    Method responsible for the partioning of the matrix

    Parameters
    ----------
    hand : int or flat array
        Height above the nearest drainage.
    flow_accumulation : int
        Flow accumulation. Represent the river as source of hazard.
    indices : indices : int array
        River cell index.
    n_gfi : float
        Expoent (<1) Calibrated for the region.
    scale_factor : float
        Scale factor.

    Returns
    -------
    gfi : float array
        geomorphic flood index array.

    '''
    row_size = len(hand)
    col_size = len(hand[0])

    boundary_row, boundary_column = divisor(row_size, col_size, division_row,
                                            division_column)

    flow_accumulation = river_accumulation(flow_accumulation, indices)

    gfi = np.zeros((row_size, col_size))

    boundary_row = np.insert(boundary_row, division_row, row_size)
    boundary_row = np.insert(boundary_row, 0, 0)
    boundary_column = np.insert(boundary_column, division_column, col_size)
    boundary_column = np.insert(boundary_column, 0, 0)

    for m in range(0, division_row + 1, 1):
        for n in range(0, division_column + 1, 1):

            mS = boundary_row[m]
            mE = boundary_row[m + 1]
            nS = boundary_column[n]
            nE = boundary_column[n + 1]

            gfi[mS:mE, nS:nE] = geomorphic_flood_index_cpu(
                hand[mS:mE, nS:nE], flow_accumulation[mS:mE, nS:nE], n_gfi,
                scale_factor, size)

    return gfi


def geomorphic_flood_index_cpu(hand,
                               river_flow_accumulation,
                               expoent,
                               scale_factor,
                               size,
                               blocks=0,
                               threads=0):
    '''
    Method responsible for the host/device data transfer

    Parameters
    ----------
    hand : int or flat array
        Height above the nearest drainage.
    river_flow_accumulation : int
        Flow accumulation. Represent the river as source of hazard.
    expoent : float
        Expoent (<1) Calibrated for the region.
    scale_factor : float
        Scale factor.
    blocks : int, optional
        Number of block of threads. The default is 0.
    threads : int, optional
        number of threads in each block. The default is 0.

    Returns
    -------
    gfi : float array
        geomorphic flood index array.

    '''
    row = len(hand)
    col = len(hand[0])
    if blocks == 0 and threads == 0:
        threads = 256
        blocks = math.ceil((row * col) / threads)

    hand = np.asarray(hand).reshape(-1)
    river_flow_accumulation = np.asarray(river_flow_accumulation).reshape(-1)

    gfi = np.zeros((row * col), dtype='float32')

    river_flow_accumulation = cuda.to_device(river_flow_accumulation)
    hand = cuda.to_device(hand)

    gfi = cuda.to_device(gfi)

    geomorphic_flood_index_gpu[blocks,
                               threads](hand, river_flow_accumulation, gfi,
                                        expoent, scale_factor, size)

    gfi = gfi.copy_to_host()
    gfi = gfi.reshape(row, col)

    return gfi


@cuda.jit
def geomorphic_flood_index_gpu(hand, river_flow_accumulation, gfi, expoent,
                               scale_factor, size):
    '''
    GPU GFI index method

    Parameters
    ----------
    hand : int or flat array
        Height above the nearest drainage.
    river_flow_accumulation : int
        Flow accumulation. Represent the river as source of hazard.
    gfi : float array
        geomorphic flood index array. Initialized as zeros.
    expoent : float
        Expoent (<1) Calibrated for the region.
    scale_factor : float
        Scale factor.

    '''
    i = cuda.grid(1)
    if i >= 0 and i < len(hand):
        if hand[i] <= -100:
            gfi[i] = -100
        else:
            gfi[i] = math.log(scale_factor * (math.pow(
                (river_flow_accumulation[i] * (size * size)), expoent)) /
                              (hand[i] + 0.01))


def ln_hl_H_calculator(hand, 
                       flow_accumulation, 
                       n_gfi, 
                       scale_factor, 
                       size, 
                       division_column=0, 
                       division_row=0):
    '''
    Method responsible for the partioning of the matrix

    Parameters
    ----------
    hand : int or flat array
        Height above the nearest drainage.
    fac : int
        Flow accumulation. 
    n_gfi : float
        Expoent (<1) Calibrated for the region.
    scale_factor : float
        Scale factor.

    Returns
    -------
    ln_hl_H : float array
        ln(hl/H) index array.

    '''
    row_size = len(hand)
    col_size = len(hand[0])

    bRow, bCol = divisor(row_size, col_size, division_row, division_column)

    lnhlh = np.zeros((row_size, col_size))
    bRow = np.insert(bRow, division_row, row_size)
    bRow = np.insert(bRow, 0, 0)
    bCol = np.insert(bCol, division_column, col_size)
    bCol = np.insert(bCol, 0, 0)

    for m in range(0, division_row + 1, 1):
        for n in range(0, division_column + 1, 1):
            mS = bRow[m]
            mE = bRow[m + 1]
            nS = bCol[n]
            nE = bCol[n + 1]

            lnhlh[mS:mE, nS:nE] = ln_hl_H_cpu(hand[mS:mE, nS:nE],
                                              flow_accumulation[mS:mE, nS:nE],
                                              n_gfi, scale_factor, size)

    return lnhlh


def ln_hl_H_cpu(hand,
                flow_accumulation,
                expoent,
                scale_factor,
                size,
                blocks=0,
                threads=0):
    '''
    Method responsible for the host/device data transfer

    Parameters
    ----------
    hand : int or flat array
        Height above the nearest drainage.
    flow_accumulation : int
        Flow accumulation. 
    expoent : float
        Expoent (<1) Calibrated for the region.
    scale_factor : float
        Scale factor.
    blocks : int, optional
        Number of block of threads. The default is 0.
    threads : int, optional
        number of threads in each block. The default is 0.

    Returns
    -------
    ln_hl_H : float array
        ln(hl/H) index array.

    '''
    row = len(hand)
    col = len(hand[0])
    if blocks == 0 and threads == 0:
        threads = 256
        blocks = math.ceil((row * col) / threads)

    hand = np.asarray(hand).reshape(-1)
    flow_accumulation = np.asarray(flow_accumulation).reshape(-1)
    lnhlh = np.zeros((row * col), dtype='float32')

    flow_accumulation = cuda.to_device(flow_accumulation)
    hand = cuda.to_device(hand)
    lnhlh = cuda.to_device(lnhlh)

    ln_hl_H_gpu[blocks, threads](hand, flow_accumulation, lnhlh, expoent,
                                 scale_factor, size)

    lnhlh = lnhlh.copy_to_host()
    lnhlh = lnhlh.reshape(row, col)

    return lnhlh


@cuda.jit
def ln_hl_H_gpu(hand, flow_accumulation, ln_hl_H, expoent, scale_factor, size):
    '''
    GPU ln(hl/HAND) index method

    Parameters
    ----------
    hand : int or flat array
        Height above the nearest drainage.
    flow_accumulation : int
        Flow accumulation. Represent the river as source of hazard.
    ln_hl_H : float array
        lnhlh index array. Initialized as zeros.
    expoent : float
        Expoent (<1) Calibrated for the region.
    scale_factor : float
        Scale factor.

    Returns 
    --------
    ln_hl_H : float array
        ln(hl/H) index array.

    '''
    i = cuda.grid(1)
    if i >= 0 and i < len(hand):
        if hand[i] <= -100:
            ln_hl_H[i] = -100
        else:
            if flow_accumulation[i] == 0:
                ln_hl_H[i] = math.log(
                    (scale_factor * math.pow(1 * (size * size), expoent)) /
                    (hand[i] + 0.01))
            else:
                ln_hl_H[i] = math.log(
                    (scale_factor *
                     math.pow(flow_accumulation[i] *
                              (size * size), expoent)) / (hand[i] + 0.01))
