# -*- coding: utf-8 -*-
from numba import cuda, jit, float32
import numpy as np
import math

from descriptools.helpers import divisor


def topographic_index_sequential(flow_accumulation, slope, px):
    '''
    Sequential method of the Topographic index calculation

    Parameters
    ---------- 
    flow_accumulation : int
        Flow accumulation. Servers as the...
    slope : float
        Highest slope to a neighbouring cell. Surrogate to...
    px : int or float
        Raster dimension size.

    Returns
    -------
    topographic_index : float
        Topographic index.

    '''
    topographic_index = np.where(
        slope == -100, -100,
        np.log((np.where(flow_accumulation == 0, 1, flow_accumulation) *
                np.power(px, 2)) / (np.tan(slope))))

    return topographic_index


@jit
def topographic_index_sequential_jit(flow_accumulation, slope, px):
    topographic_index = np.zeros(slope.shape, dtype=float32)

    for i in range(0, len(slope), 1):
        for j in range(0, len(slope[0]), 1):
            if slope[i, j] == -100:
                topographic_index[i, j] = -100
            else:
                if flow_accumulation[i, j] == 0:
                    topographic_index[i, j] = np.log(
                        (1 * px * px) / (np.tan(slope[i, j]) + 0.01))
                else:
                    topographic_index[i, j] = np.log(
                        (flow_accumulation[i, j] * px * px) /
                        (np.tan(slope[i, j]) + 0.01))

    return topographic_index


@jit
def modified_topographic_index_sequential_jit(flow_accumulation, slope, px,
                                              expoent):
    modified_topographic_index = np.zeros(slope.shape, dtype=float32)

    for i in range(0, len(slope), 1):
        for j in range(0, len(slope[0]), 1):
            if slope[i, j] == -100:
                modified_topographic_index[i, j] = -100
            else:
                if flow_accumulation[i, j] == 0:
                    modified_topographic_index[i, j] = np.log(
                        np.power(1 * px * px, expoent) /
                        (np.tan(slope[i, j]) + 0.01))
                else:
                    modified_topographic_index[i, j] = np.log(
                        np.power(flow_accumulation[i, j] * px * px, expoent) /
                        (np.tan(slope[i, j]) + 0.01))

    return modified_topographic_index


def modified_topographic_index_sequential(flow_accumulation, slope, px,
                                          expoent):
    '''
    Sequential method of the Modified Topographic Index calculation

    Parameters
    ----------
    flow_accumulation : int
        Flow accumulation. Servers as the...
    slope : float
        Highest slope to a neighbouring cell. Surrogate to...
    px : int or float
        Raster dimension size.
    expoent : float
        Expoent (<1) calibrated for the area

    Returns
    -------
    modified_topographic_index : float
        Modified Topographic index.
    '''

    modified_topographic_index = np.where(
        slope == -100, -100,
        np.log(
            np.power((np.where(flow_accumulation == 0, 1, flow_accumulation) *
                      np.power(px, 2)), expoent) / (np.tan(slope))))

    return modified_topographic_index


def topographic_index(flow_accumulation,
                      slope,
                      px,
                      n_top,
                      div_col=0,
                      div_row=0):
    '''
    Method responsible for the partioning of the matrix

    Parameters
    ----------
    flow_accumulation : int
        Flow accumulation. Servers as the...
    slope : float
        Highest slope to a neighbouring cell. Surrogate to...
    px : int or float
        Raster dimension size.
    n_top : float
        Expoent (<1) calibrated for the area
    div_col : int, optional
        Number of vertical divisions. The default is 0.
    div_row : int, optional
        Number of horizontal divisions. The default is 0.

    Returns
    -------
    topographic_index : float
        Topographic index..
    modified_topographic_index : float
        Modified Topographic index.

    '''
    row_size = len(flow_accumulation)
    col_size = len(flow_accumulation[0])

    bRow, bCol = divisor(row_size, col_size, div_row, div_col)

    topographic_index = np.zeros((row_size, col_size))
    modified_topographic_index = np.zeros((row_size, col_size))

    bRow = np.insert(bRow, div_row, row_size)
    bRow = np.insert(bRow, 0, 0)
    bCol = np.insert(bCol, div_col, col_size)
    bCol = np.insert(bCol, 0, 0)

    for m in range(0, div_row + 1, 1):
        for n in range(0, div_col + 1, 1):

            mS = bRow[m]
            mE = bRow[m + 1]
            nS = bCol[n]
            nE = bCol[n + 1]

            topographic_index[mS:mE, nS:nE], modified_topographic_index[
                mS:mE,
                nS:nE] = topographic_index_cpu(flow_accumulation[mS:mE, nS:nE],
                                               slope[mS:mE, nS:nE], px, n_top)

    return topographic_index, modified_topographic_index


def topographic_index_cpu(flow_accumulation,
                          slope,
                          px,
                          expoent,
                          blocks=0,
                          threads=0):
    '''
    Method responsible for the host/device data transfer

    Parameters
    ----------
    flow_accumulation : int
        Flow accumulation. Servers as the...
    slope : float
        Highest slope to a neighbouring cell. Surrogate to...
    px : int or float
        Raster dimension size.
    expoent : float
        Expoent (<1) calibrated for the area
    blocks : int, optional
        Number of block of threads. The default is 0.
    threads : int, optional
        number of threads in each block. The default is 0.

    Returns
    -------
    topographic_index : float
        Topographic index.
    modified_topographic_index : float
        Modified Topographic index.

    '''
    row = len(flow_accumulation)
    col = len(flow_accumulation[0])
    if blocks == 0 and threads == 0:
        threads = 256
        blocks = math.ceil((row * col) / threads)

    flow_accumulation = np.asarray(flow_accumulation).reshape(-1)
    slope = np.asarray(slope).reshape(-1)
    topographic_index = np.zeros((row * col), dtype='float32')
    modified_topographic_index = np.zeros((row * col), dtype='float32')

    flow_accumulation = cuda.to_device(flow_accumulation)
    slope = cuda.to_device(slope)
    topographic_index = cuda.to_device(topographic_index)
    modified_topographic_index = cuda.to_device(modified_topographic_index)

    topographic_index_gpu[blocks, threads](flow_accumulation, slope,
                                           topographic_index, px)
    modified_topographic_index_gpu[blocks, threads](flow_accumulation, slope,
                                                    modified_topographic_index,
                                                    px, expoent)

    topographic_index = topographic_index.copy_to_host()
    modified_topographic_index = modified_topographic_index.copy_to_host()

    topographic_index = topographic_index.reshape(row, col)
    modified_topographic_index = modified_topographic_index.reshape(row, col)

    return topographic_index, modified_topographic_index


@cuda.jit
def topographic_index_gpu(flow_accumulation, slope, topographic_index, px):
    '''
    GPU Topographic index method

    Parameters
    ----------
    flow_accumulation : int
        Flow accumulation. Servers as the...
    slope : float
        Highest slope to a neighbouring cell. Surrogate to...
    topographic_index : float
        Topographic index.
    px : int or float
        Raster dimension size.

    '''
    i = cuda.grid(1)
    if i >= 0 and i < len(slope):
        if flow_accumulation[i] <= -100:
            topographic_index[i] = -100
        else:
            if flow_accumulation[i] == 0:
                topographic_index[i] = math.log(
                    (1 * (px*px)) / (math.tan(slope[i] + 0.01)))
            else:
                topographic_index[i] = math.log(
                    (flow_accumulation[i] * (px*px)) /
                    (math.tan(slope[i] + 0.01)))


@cuda.jit
def modified_topographic_index_gpu(flow_accumulation, slope,
                                   modified_topographic_index, px, expoent):
    '''
    GPU Modified Topographic index method

    Parameters
    ----------
    flow_accumulation : int
        Flow accumulation. Servers as the...
    slope : float
        Highest slope to a neighbouring cell. Surrogate to...
    modified_topographic_index : float
        Modified Topographic index.
    px : int or float
        Raster dimension size.
    expoent : float
        Expoent (<1) calibrated for the area

    '''
    i = cuda.grid(1)
    if i >= 0 and i < len(slope):
        if flow_accumulation[i] <= -100:
            modified_topographic_index[i] = -100
        else:
            if flow_accumulation[i] == 0:
                modified_topographic_index[i] = math.log(
                    ((1 * (px*px))**expoent) / (math.tan(slope[i] + 0.01)))
            else:
                modified_topographic_index[i] = math.log(
                    ((flow_accumulation[i] * (px*px))**expoent) /
                    (math.tan(slope[i] + 0.01)))
