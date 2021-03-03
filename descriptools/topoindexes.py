# -*- coding: utf-8 -*-
from numba import cuda, jit, float32
import numpy as np
import math


def divisor(len_row, len_col, div_row, div_col):
    bCol = np.array([], dtype=int)
    bRow = np.array([], dtype=int)

    for i in range(0, div_row, 1):
        bRow = np.append(bRow, [math.floor((i + 1) * len_row / (div_row + 1))])
    for i in range(0, div_col, 1):
        bCol = np.append(bCol, [math.floor((i + 1) * len_col / (div_col + 1))])

    return bRow, bCol


def topoindex_sequential(fac, slope, px):
    '''
    Sequential method of the Topographic index calculation

    Parameters
    ---------- 
    fac : int
        Flow accumulation. Servers as the...
    slope : float
        Highest slope to a neighbouring cell. Surrogate to...
    px : int or float
        Raster dimension size.

    Returns
    -------
    tp : float
        Topographic index.

    '''
    tp = np.where(
        slope == -100, -100,
        np.log(
            (np.where(fac == 0, 1, fac) * np.power(px, 2)) / (np.tan(slope))))

    # tp = np.where

    return tp


@jit
def topoindex_sequential_jit(fac, slope, px):
    tp = np.zeros(slope.shape, dtype=float32)

    for i in range(0, len(slope), 1):
        for j in range(0, len(slope[0]), 1):
            if slope[i, j] == -100:
                tp[i, j] = -100
            else:
                if fac[i, j] == 0:
                    tp[i, j] = np.log(
                        (1 * px * px) / (np.tan(slope[i, j]) + 0.0001))
                    # lnhlh[i,j] = 1
                else:
                    tp[i, j] = np.log(
                        (fac[i, j] * px * px) / (np.tan(slope[i, j]) + 0.0001))
                    # tp[i,j] = math.log(((fac[i]*px**2)**n)/(math.tan(slope[i])))
                    # lnhlh[i,j] = np.log(b*(np.power(fac[i,j], n))/hand[i,j])
                    # lnhlh[i,j] = 2

    return tp


@jit
def modtopoindex_sequential_jit(fac, slope, px, n):
    mtp = np.zeros(slope.shape, dtype=float32)

    for i in range(0, len(slope), 1):
        for j in range(0, len(slope[0]), 1):
            if slope[i, j] == -100:
                mtp[i, j] = -100
            else:
                if fac[i, j] == 0:
                    # mtp[i,j] = np.log(np.power((1*px*px),n))
                    mtp[i, j] = np.log(
                        np.power(1 * px * px, n) /
                        (np.tan(slope[i, j]) + 0.0001))
                    # lnhlh[i,j] = 1
                else:
                    mtp[i, j] = np.log(
                        np.power(fac[i, j] * px * px, n) /
                        (np.tan(slope[i, j]) + 0.0001))
                    # tp[i,j] = math.log(((fac[i]*px**2)**n)/(math.tan(slope[i])))
                    # lnhlh[i,j] = np.log(b*(np.power(fac[i,j], n))/hand[i,j])
                    # lnhlh[i,j] = 2

    return mtp


def modtopoindex_sequential(fac, slope, px, n):
    '''
    Sequential method of the Modified Topographic Index calculation

    Parameters
    ----------
    fac : int
        Flow accumulation. Servers as the...
    slope : float
        Highest slope to a neighbouring cell. Surrogate to...
    px : int or float
        Raster dimension size.
    n : float
        Expoent (<1) calibrated for the area

    Returns
    -------
    mtp : float
        Modified Topographic index.
    '''

    mtp = np.where(
        slope == -100, -100,
        np.log(
            np.power((np.where(fac == 0, 1, fac) * np.power(px, 2)), n) /
            (np.tan(slope))))

    return mtp


def topoIndex(fac, slope, px, ntop, div_col=0, div_row=0):
    '''
    Method responsible for the partioning of the matrix

    Parameters
    ----------
    fac : int
        Flow accumulation. Servers as the...
    slope : float
        Highest slope to a neighbouring cell. Surrogate to...
    px : int or float
        Raster dimension size.
    n : float
        Expoent (<1) calibrated for the area
    div_col : int, optional
        Number of vertical divisions. The default is 0.
    div_row : int, optional
        Number of horizontal divisions. The default is 0.

    Returns
    -------
    TI : float
        Topographic index..
    MTI : float
        Modified Topographic index.

    '''
    row_size = len(fac)
    col_size = len(fac[0])

    # div_col = 0
    # div_row = 0

    bRow, bCol = divisor(row_size, col_size, div_row, div_col)

    TI = np.zeros((row_size, col_size))
    MTI = np.zeros((row_size, col_size))

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

            # print(dem[mS:mE,nS:nE])

            TI[mS:mE,
               nS:nE], MTI[mS:mE,
                           nS:nE] = topoIndex_host(fac[mS:mE, nS:nE],
                                                   slope[mS:mE,
                                                         nS:nE], px, ntop)
            # TI[mS:mE,nS:nE] = topoIndex_host(fac[mS:mE,nS:nE],
            #                                     slope[mS:mE,nS:nE], px, n)

    return TI, MTI
    # return TI


def topoIndex_host(fac, slope, px, n, blocks=0, threads=0):
    '''
    Method responsible for the host/device data transfer

    Parameters
    ----------
    fac : int
        Flow accumulation. Servers as the...
    slope : float
        Highest slope to a neighbouring cell. Surrogate to...
    px : int or float
        Raster dimension size.
    n : float
        Expoent (<1) calibrated for the area
    blocks : int, optional
        Number of block of threads. The default is 0.
    threads : int, optional
        number of threads in each block. The default is 0.

    Returns
    -------
    TI : float
        Topographic index.
    MTI : float
        Modified Topographic index.

    '''
    row = len(fac)
    col = len(fac[0])
    if blocks == 0 and threads == 0:
        threads = 256
        blocks = math.ceil((row * col) / threads)

    fac = np.asarray(fac).reshape(-1)
    slope = np.asarray(slope).reshape(-1)
    TI = np.zeros((row * col), dtype='float32')
    MTI = np.zeros((row * col), dtype='float32')

    fac = cuda.to_device(fac)
    slope = cuda.to_device(slope)
    TI = cuda.to_device(TI)
    MTI = cuda.to_device(MTI)

    topoIndex_device[blocks, threads](fac, slope, TI, px)
    topoIndexMod_device[blocks, threads](fac, slope, MTI, px, n)

    TI = TI.copy_to_host()
    MTI = MTI.copy_to_host()

    TI = TI.reshape(row, col)
    MTI = MTI.reshape(row, col)

    return TI, MTI
    # return TI


@cuda.jit
def topoIndex_device(fac, slope, TI, px):
    '''
    GPU Topographic index method

    Parameters
    ----------
    fac : int
        Flow accumulation. Servers as the...
    slope : float
        Highest slope to a neighbouring cell. Surrogate to...
    TI : float
        Topographic index.
    px : int or float
        Raster dimension size.

    '''
    i = cuda.grid(1)
    if i >= 0 and i < len(slope):
        if fac[i] <= -100:
            TI[i] = -100
        else:
            if fac[i] == 0:
                TI[i] = math.log((1 * px**2) / (math.tan(slope[i] + 0.0001)))
            else:
                TI[i] = math.log(
                    (fac[i] * px**2) / (math.tan(slope[i] + 0.0001)))


@cuda.jit
def topoIndexMod_device(fac, slope, MTI, px, n):
    '''
    GPU Modified Topographic index method

    Parameters
    ----------
    fac : int
        Flow accumulation. Servers as the...
    slope : float
        Highest slope to a neighbouring cell. Surrogate to...
    MTI : float
        Modified Topographic index.
    px : int or float
        Raster dimension size.
    n : float
        Expoent (<1) calibrated for the area

    '''
    i = cuda.grid(1)
    if i >= 0 and i < len(slope):
        if fac[i] <= -100:
            MTI[i] = -100
        else:
            if fac[i] == 0:
                # MTI[i] = math.log((1.0*px*px)**n)
                # MTI[i] = n
                MTI[i] = math.log(
                    ((1 * px**2)**n) / (math.tan(slope[i] + 0.0001)))
            else:
                MTI[i] = math.log(
                    ((fac[i] * px**2)**n) / (math.tan(slope[i] + 0.0001)))
