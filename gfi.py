# -*- coding: utf-8 -*-
"""
gfi and ln(hl/H)

@author: Acer
"""
from numba import cuda, jit, float32
# from pysheds.grid import Grid
import numpy as np
# from matplotlib import pyplot as plt
import math

def divisor(len_row,len_col,div_row,div_col):
    bCol = np.array([], dtype=int)
    bRow = np.array([], dtype=int)
    
    for i in range(0,div_row,1):
        bRow = np.append(bRow,[math.floor((i+1)*len_row/(div_row+1))])
    for i in range(0,div_col,1):
        bCol = np.append(bCol,[math.floor((i+1)*len_col/(div_col+1))])
    
    return bRow, bCol

def gfi_sequential(hand,fac,indices,n,b):
    '''
    Sequential method for the GFI index

    Parameters
    ----------
    hand : int or flat array
        Height above the nearest drainage.
    fac : int
        Flow accumulation. Represent the river as source of hazard.
    indices : int array
        River cell index.
    n : float
        Expoent (<1) Calibrated for the region.
    b : float
        Scale factor.

    Returns
    -------
    gfi : float array
        geomorphic flood index array.

    '''
    facr = river_accumulation(fac, indices)
    
    gfi = np.where(hand == -100,-100,np.where(hand==0,0,np.log(b*(np.power(np.where(facr==0,1,facr), n))/hand)))
    
    return gfi

@jit
def gfi_sequential_jit(hand,fac,indices,n,b):
    facr = river_accumulation(fac, indices)
    gfi = np.zeros(hand.shape,dtype=float32)
    
    for i in range(0,len(hand),1):
        for j in range(0,len(hand[0]),1):
            if hand[i,j] == -100:
                gfi[i,j] = -100
            # elif hand[i,j] == 0:
            #     gfi[i,j] = np.log(b*(np.power(facr[i,j], n))/(hand[i,j]+0.01))
            else:
                # if facr[i,j] == 0:
                #     gfi[i,j] = np.log(b*(np.power(1, n))/hand[i,j])
                #     # lnhlh[i,j] = 1
                # else:
                    gfi[i,j] = np.log(b*(np.power(facr[i,j], n))/(hand[i,j]+0.01))
                    # lnhlh[i,j] = 2
                
    return gfi

@jit
def lnhlh_sequential_jit(hand,fac,n,b):
    
    lnhlh = np.zeros(hand.shape,dtype=float32)
    
    for i in range(0,len(hand),1):
        for j in range(0,len(hand[0]),1):
            if hand[i,j] == -100:
                lnhlh[i,j] = -100
            elif hand[i,j] == 0:
                lnhlh[i,j] = 0
            else:
                if fac[i,j] == 0:
                    lnhlh[i,j] = np.log(b*(np.power(1, n))/(hand[i,j]+0.01))
                    # lnhlh[i,j] = 1
                else:
                    lnhlh[i,j] = np.log(b*(np.power(fac[i,j], n))/(hand[i,j]+0.01))
                    # lnhlh[i,j] = 2
                
    return lnhlh

def lnhlh_sequential(hand,fac,n,b):
    '''
    Sequential method for the ln(hl/HAND) index

    Parameters
    ----------
    hand : int or flat array
        Height above the nearest drainage.
    fac : int
        Flow accumulation. Represent the upslope area as source of hazard.
    n : float
        Expoent (<1) Calibrated for the region.
    b : float
        Scale factor.

    Returns
    -------
    gfi : float array
        geomorphic flood index array.

    '''
    lnhlh = np.where(hand == -100,-100,np.where(hand==0,0,np.log(b*(np.power(np.where(fac==0,1,fac), n))/hand)))
    #np.where(fac==0,1,fac)
    return lnhlh

@jit
def river_accumulation(fac,indices):
    '''
    Method that return the array with the river cell flow accumulation.

    Parameters
    ----------
    fac : int
        Flow accumulation. 
    indices : indices : int array
        River cell index.

    Returns
    -------
    fac_river : int
        Flow accumulation of the river cell. Represent the river as source of hazard.

    '''
    row, col = fac.shape
    fac = np.asarray(fac).reshape(-1)
    indices = np.asarray(indices).reshape(-1)
    fac_river = np.zeros(row*col, float32)
    
    # fac_river = np.where(fac != -100,fac[indices],-100)
    fac_river = np.where(indices != -100,fac[indices],fac[0])
    
    fac_river = fac_river.reshape(row,col)
    
    return fac_river

def river_accumulation_old(fac,indices): 
    fac_river = np.zeros((len(fac),len(fac[0])))
    for i in range(0,len(fac),1):
        for j in range(0,len(fac[0]),1):
            ii = math.floor(indices[i,j]/len(indices[0]))
            jj = indices[i,j]%len(indices[0])
            jj = jj.astype('int')
            fac_river[i,j] = fac[ii,jj]
    return fac_river

def gfi_calculator(hand,fac,indices,ngfi,b,size):
    '''
    Method responsible for the partioning of the matrix

    Parameters
    ----------
    hand : int or flat array
        Height above the nearest drainage.
    fac : int
        Flow accumulation. Represent the river as source of hazard.
    indices : indices : int array
        River cell index.
    n : float
        Expoent (<1) Calibrated for the region.
    b : float
        Scale factor.

    Returns
    -------
    gfi : float array
        geomorphic flood index array.

    '''
    row_size = len(hand)
    col_size = len(hand[0])
    
    div_col = 2
    div_row = 2
    
    bRow, bCol = divisor(row_size,col_size,div_row,div_col)
    
    fac = river_accumulation(fac, indices)
    
    gfi = np.zeros((row_size,col_size))
    
    bRow = np.insert(bRow,div_row,row_size)
    bRow = np.insert(bRow,0,0)
    bCol = np.insert(bCol,div_col,col_size)
    bCol = np.insert(bCol,0,0)
    
    for m in range(0,div_row+1,1):
        for n in range(0,div_col+1,1):
            
            mS = bRow[m]
            mE = bRow[m+1]
            nS = bCol[n]
            nE = bCol[n+1]

            gfi[mS:mE,nS:nE]= gfi_host(hand[mS:mE,nS:nE], 
                                       fac[mS:mE,nS:nE], ngfi, b,size)
            
    return gfi
    
def gfi_host(hand,fac_rio,n,b,size,blocks=0,threads=0):
    '''
    Method responsible for the host/device data transfer

    Parameters
    ----------
    hand : int or flat array
        Height above the nearest drainage.
    fac : int
        Flow accumulation. Represent the river as source of hazard.
    n : float
        Expoent (<1) Calibrated for the region.
    b : float
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
        blocks = math.ceil((row*col)/threads)

    hand = np.asarray(hand).reshape(-1)
    fac_rio = np.asarray(fac_rio).reshape(-1)
    
    gfi = np.zeros((row*col), dtype='float32')
    
    fac_rio = cuda.to_device(fac_rio)
    hand = cuda.to_device(hand)

    gfi = cuda.to_device(gfi)
    
    gfi_device[blocks,threads](hand,fac_rio,gfi,n,b,size)
    
    gfi = gfi.copy_to_host()
    gfi = gfi.reshape(row,col)

    return gfi
  
@cuda.jit
def gfi_device(hand,facr,gfi,n,b,size):
    '''
    GPU GFI index method

    Parameters
    ----------
    hand : int or flat array
        Height above the nearest drainage.
    fac : int
        Flow accumulation. Represent the river as source of hazard.
    gfi : float array
        geomorphic flood index array. Initialized as zeros.
    n : float
        Expoent (<1) Calibrated for the region.
    b : float
        Scale factor.

    '''
    i = cuda.grid(1)
    if i >= 0 and i < len(hand):
        if hand[i] <= -100:
            gfi[i] = -100
        else:
            # if hand[i] == 0:
                # gfi[i] = 0
            # else:
                gfi[i] = math.log(b*(math.pow((facr[i] * (size*size)), n))/(hand[i]+0.01))
                # gfi[i] = math.log(b*(math.pow(facr[i], n))/(hand[i]))
            
def lnhlh_calculator(hand,fac,ngfi,b,size):
    '''
    Method responsible for the partioning of the matrix

    Parameters
    ----------
    hand : int or flat array
        Height above the nearest drainage.
    fac : int
        Flow accumulation. 
    n : float
        Expoent (<1) Calibrated for the region.
    b : float
        Scale factor.

    Returns
    -------
    lnhlh : float array
        lnhlh index array.

    '''
    row_size = len(hand)
    col_size = len(hand[0])
    
    div_col = 0
    div_row = 0
    
    bRow, bCol = divisor(row_size,col_size,div_row,div_col)
    
    lnhlh = np.zeros((row_size,col_size))   
    bRow = np.insert(bRow,div_row,row_size)
    bRow = np.insert(bRow,0,0)
    bCol = np.insert(bCol,div_col,col_size)
    bCol = np.insert(bCol,0,0)
    
    for m in range(0,div_row+1,1):
        for n in range(0,div_col+1,1):          
            mS = bRow[m]
            mE = bRow[m+1]
            nS = bCol[n]
            nE = bCol[n+1]

            lnhlh[mS:mE,nS:nE]= lnhlh_host(hand[mS:mE,nS:nE], 
                                       fac[mS:mE,nS:nE], ngfi, b,size)
            
    return lnhlh
    
def lnhlh_host(hand,fac,n,b,size,blocks=0,threads=0):
    '''
    Method responsible for the host/device data transfer

    Parameters
    ----------
    hand : int or flat array
        Height above the nearest drainage.
    fac : int
        Flow accumulation. 
    n : float
        Expoent (<1) Calibrated for the region.
    b : float
        Scale factor.
    blocks : int, optional
        Number of block of threads. The default is 0.
    threads : int, optional
        number of threads in each block. The default is 0.

    Returns
    -------
    lnhlh : float array
        lnhlh index array.

    '''
    row = len(hand)
    col = len(hand[0])
    if blocks == 0 and threads == 0:
        threads = 256
        blocks = math.ceil((row*col)/threads)
    
    hand = np.asarray(hand).reshape(-1)
    fac = np.asarray(fac).reshape(-1)
    lnhlh = np.zeros((row*col), dtype='float32')
    
    fac = cuda.to_device(fac)
    hand = cuda.to_device(hand)
    lnhlh = cuda.to_device(lnhlh)
    
    lnhlh_device[blocks,threads](hand,fac,lnhlh,n,b,size)
    
    lnhlh = lnhlh.copy_to_host()
    lnhlh = lnhlh.reshape(row,col)
    
    return lnhlh

@cuda.jit
def lnhlh_device(hand,fac,lnhlh,n,b,size):
    '''
    GPU ln(hl/HAND) index method

    Parameters
    ----------
    hand : int or flat array
        Height above the nearest drainage.
    fac : int
        Flow accumulation. Represent the river as source of hazard.
    lnhlh : float array
        lnhlh index array. Initialized as zeros.
    n : float
        Expoent (<1) Calibrated for the region.
    b : float
        Scale factor.

    '''
    i = cuda.grid(1)
    if i >= 0 and i < len(hand):
        if hand[i] <= -100:
            lnhlh[i] = -100
        # elif hand[i] == 0:
        #     lnhlh[i] = 0
        else:
            if fac[i] == 0:
                lnhlh[i] =  math.log((b*math.pow(1* (size*size), n))/(hand[i]+0.01))
            else:
                lnhlh[i] =  math.log((b*math.pow(fac[i]* (size*size), n))/(hand[i]+0.01))

    
    
    
    
    