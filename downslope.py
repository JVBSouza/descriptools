# -*- coding: utf-8 -*-
"""
downslope

@author: Acer
"""
from numba import cuda, jit, float32
# from pysheds.grid import Grid
import numpy as np
# from matplotlib import pyplot as plt
import math

import time

def divisor(len_row,len_col,div_row,div_col):
    bCol = np.array([], dtype=int)
    bRow = np.array([], dtype=int)
    
    for i in range(0,div_row,1):
        bRow = np.append(bRow,[math.floor((i+1)*len_row/(div_row+1))])
    for i in range(0,div_col,1):
        bCol = np.append(bCol,[math.floor((i+1)*len_col/(div_col+1))])
    
    return bRow, bCol

def downslope_sequential(dem,fdr,px,dif,downslope=np.array([])):
    '''
    Downslope sequential method. Also responsible for fixing cells that could
    not be simulated in the gpu.

    Parameters
    ----------
    dem : int or float
        Digital elevation model.
    fdr : int
        Flow direction.
    px : int or float
        Raster pixel dimension.
    dif : int
        Elevation difference.
    downslope : Downslope float array, optional
        Downslope array. The default is np.array([]).

    Returns
    -------
    downslope : float array
        Downslope index array.

    '''
    new = 0
    if downslope.size == 0:
        downslope = np.zeros([len(dem),len(dem[0])],dtype='float32')
        new = 1
    
    for i in range(622,len(downslope),1):
        for j in range(2020,len(downslope[0]),1):
            if dem[i,j] == -100:
                downslope[i,j] = -100
                continue
            elif new == 0 and downslope[i,j] != -50:
                continue
            else:
                y = i
                x = j
                dist = 0
                loop = 0
                isnan = 0
                while(dem[i,j] - dem[y,x] < dif):
                    if y == 0 and (fdr[y,x]==32 or fdr[y,x]==64 or fdr[y,x]==128):
                        isnan = 1
                        break
                    elif y ==len(downslope)-1  and (fdr[y,x]==2 or fdr[y,x]==4 or fdr[y,x]==8):
                        isnan = 1
                        break
                    elif x == 0  and (fdr[y,x]==32 or fdr[y,x]== 16 or fdr[y,x]== 8):
                        isnan = 1
                        break
                    elif x ==len(downslope[0])-1  and (fdr[y,x]== 128 or fdr[y,x]== 1 or fdr[y,x]== 2):
                        isnan = 1
                        break
                    
                    #alguma condição de fronteira aqui                    
                    if fdr[y,x] == 1:
                        if dem[y,x+1] == -100:
                            isnan = 2
                            break
                        x += 1
                        dist += px
                    elif fdr[y,x] == 2:
                        if dem[y+1,x+1] == -100:
                            isnan = 3
                            break
                        x += 1
                        y += 1
                        dist += px*math.sqrt(2.0) #Arrumar raiz aqui
                    elif fdr[y,x] == 4:
                        if dem[y+1,x] == -100:
                            isnan = 4
                            break
                        y += 1
                        dist += px
                    elif fdr[y,x] == 8:
                        if dem[y+1,x-1] == -100:
                            isnan = 5
                            break
                        x -= 1
                        y += 1
                        dist += px*math.sqrt(2.0) #Arrumar raiz aqui
                    elif fdr[y,x] == 16:
                        if dem[y,x-1] == -100:
                            isnan = 6
                            break
                        x -= 1
                        dist += px
                    elif fdr[y,x] == 32:
                        if dem[y-1,x-1] == -100:
                            isnan = 7
                            break
                        x -= 1
                        y -= 1
                        dist += px*math.sqrt(2.0) #Arrumar raiz aqui
                    elif fdr[y,x] == 64:
                        if dem[y-1,x] == -100:
                            isnan = 8
                            break
                        y -= 1
                        dist += px
                    elif fdr[y,x] == 128:
                        if dem[y-1,x+1] == -100:
                            isnan = 9
                            break
                        x += 1
                        y -= 1
                        dist += px*math.sqrt(2.0) #Arrumar raiz aqui
                    elif fdr[y,x] == -100:
                        isnan = 10
                        break
                    
                    if y >= len(downslope):
                        y -= 1
                        if x >= len(downslope[0]):
                            x -= 1
                        break
                    elif x >= len(downslope[0]):
                        x -= 1
                        if y >= len(downslope):
                            y -= 1
                        break
                    
                    if dem[y,x] == -100:
                        isnan = 11
                        break
                
                    loop += 1

                    if loop == 500:
                        break;
                
                # if isnan == 1:
                #     downslope[i,j] = -100
                if isnan > 1:
                    downslope[i,j] = -100
                else:
                    downslope[i,j] = (dem[i,j] - dem[y,x])/dist
    
    return downslope

# @njit
@jit
def downslope_sequential_jit(dem,fdr,px,dif,downslope=np.array([[],[]],'float32')):
# def downslope_sequential_jit(dem,fdr,px,dif,downslope=np.array([])):
    '''
    Downslope sequential method. Also responsible for fixing cells that could
    not be simulated in the gpu.

    Parameters
    ----------
    dem : int or float
        Digital elevation model.
    fdr : int
        Flow direction.
    px : int or float
        Raster pixel dimension.
    dif : int
        Elevation difference.
    downslope : Downslope float array, optional
        Downslope array. The default is np.array([]).

    Returns
    -------
    downslope : float array
        Downslope index array.

    '''
    new = 0
    if downslope.size == 0:
        # downslope = np.zeros([len(dem),len(dem[0])],dtype=float32)
        # downslope = np.zeros((len(dem),len(dem[0])))
        downslope = np.zeros(dem.shape,dtype=float32)
        # downslope = np.zeros(dem.shape,dtype=float32)
        new = 1
    
    for i in range(0,len(downslope),1):
        for j in range(0,len(downslope[0]),1):
            if dem[i,j] == -100:
                downslope[i,j] = -100
                continue
            elif new == 0 and downslope[i,j] != -50:
                continue
            else:
                y = i
                x = j
                dist = 0
                loop = 0
                isnan = 0
                while(dem[i,j] - dem[y,x] < dif):
                    if y == 0 and (fdr[y,x]==32 or fdr[y,x]==64 or fdr[y,x]==128):
                        isnan = 1
                        break
                    elif y ==len(downslope)-1  and (fdr[y,x]==2 or fdr[y,x]==4 or fdr[y,x]==8):
                        isnan = 1
                        break
                    elif x == 0  and (fdr[y,x]==32 or fdr[y,x]== 16 or fdr[y,x]== 8):
                        isnan = 1
                        break
                    elif x ==len(downslope[0])-1  and (fdr[y,x]== 128 or fdr[y,x]== 1 or fdr[y,x]== 2):
                        isnan = 1
                        break
                    
                    #alguma condição de fronteira aqui                    
                    if fdr[y,x] == 1:
                        if dem[y,x+1] == -100:
                            isnan = 1
                            break
                        x += 1
                        dist += px
                    elif fdr[y,x] == 2:
                        if dem[y+1,x+1] == -100:
                            isnan = 1
                            break
                        x += 1
                        y += 1
                        dist += px*math.sqrt(2.0) #Arrumar raiz aqui
                    elif fdr[y,x] == 4:
                        if dem[y+1,x] == -100:
                            isnan = 1
                            break
                        y += 1
                        dist += px
                    elif fdr[y,x] == 8:
                        if dem[y+1,x-1] == -100:
                            isnan = 1
                            break
                        x -= 1
                        y += 1
                        dist += px*math.sqrt(2.0) #Arrumar raiz aqui
                    elif fdr[y,x] == 16:
                        if dem[y,x-1] == -100:
                            isnan = 1
                            break
                        x -= 1
                        dist += px
                    elif fdr[y,x] == 32:
                        if dem[y-1,x-1] == -100:
                            isnan = 1
                            break
                        x -= 1
                        y -= 1
                        dist += px*math.sqrt(2.0) #Arrumar raiz aqui
                    elif fdr[y,x] == 64:
                        if dem[y-1,x] == -100:
                            isnan = 1
                            break
                        y -= 1
                        dist += px
                    elif fdr[y,x] == 128:
                        if dem[y-1,x+1] == -100:
                            isnan = 1
                            break
                        x += 1
                        y -= 1
                        dist += px*math.sqrt(2.0) #Arrumar raiz aqui
                    elif fdr[y,x] == -100:
                        isnan = 1
                        break
                    
                    # if dem[y,x] == -100:
                        # 
                    
                    if y >= len(downslope):
                        y -= 1
                        if x >= len(downslope[0]):
                            x -= 1
                        break
                    elif x >= len(downslope[0]):
                        x -= 1
                        if y >= len(downslope):
                            y -= 1
                        break
                    
                    if dem[y,x] == -100:
                        isnan = 1
                        break
                
                    loop += 1

                    if loop == 5000:
                        break;
                
                if isnan == 1:
                    # downslope[i,j] = -100
                    # downslope[i,j] = 0
                    if dist == 0:
                        # print(y)
                        # print(x)
                        downslope[i,j] = 0
                    else:
                        downslope[i,j] = (dem[i,j] - dem[y,x])/dist
                    # downslope[i,j] = (dem[i,j] - dem[y,x])/dist
                else:
                    downslope[i,j] = (dem[i,j] - dem[y,x])/dist
    
    return downslope

def downsloper(dem, fdr, px, dif, div_col=0, div_row=0):
    '''
    Method responsible for the partioning of the matrix for the downslope.

    Parameters
    ----------
    dem : int or float
        Digital elevation model.
    fdr : int
        Flow direction.
    px : int or float
        Raster pixel dimension.
    dif : int
        Elevation difference.
    div_col : int, optional
        Number of vertical divisions. The default is 0.
    div_row : int, optional
        Number of horizontal divisions. The default is 0.

    Returns
    -------
    downslope : float array
        Downslope index array.

    '''
    row_size = len(dem)
    col_size = len(dem[0])
    
    bRow, bCol = divisor(row_size,col_size,div_row,div_col)
    
    downslope = np.zeros((row_size,col_size), dtype='float32')
    
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
            
            downslope[mS:mE,nS:nE]= downslope_host(dem[mS:mE,nS:nE], 
                                                   fdr[mS:mE,nS:nE], px, dif)
    #Aqui função pra calcular depois da gpu   
    downslope = downslope_sequential_jit(dem, fdr, px, dif, downslope)

    return downslope
    
def downslope_host(dem,fdr,px,dif,blocks=0,threads=0):
    '''
    Method responsible for the host/device data transfer

    Parameters
    ----------
    dem : int or float
        Digital elevation model.
    fdr : int
        Flow direction.
    px : int or float
        Raster pixel dimension.
    dif : int
        Elevation difference.
    blocks : int, optional
        Number of block of threads. The default is 0.
    threads : int, optional
        number of threads in each block. The default is 0.

    Returns
    -------
    downslope : float array
        Downslope index array.

    '''
    row = len(dem)
    col = len(dem[0]) 
    
    if blocks == 0 and threads == 0:
        threads = 256
        blocks = math.ceil((row*col)/threads)
        #concertar isso aqui em!!!!!!!!!!1
    # print(dem)
    dem = np.asarray(dem).reshape(-1)
    # print(dem)
    fdr = np.asarray(fdr).reshape(-1)
    downslope = np.zeros((row*col))
    
    dem = cuda.to_device(dem)
    fdr = cuda.to_device(fdr)
    downslope = cuda.to_device(downslope)
    
    downslope_device[blocks,threads](dem, fdr, downslope, px, dif, col, row)
    
    downslope = downslope.copy_to_host()
    
    # print(downslope)
    # 
    downslope = downslope.reshape(row,col)
    
    return downslope

# @cuda.jit(debug=True)
@cuda.jit
def downslope_device(dem,fdr,downslope,px,dif,col,row):
    '''
    GPU Downslope index method

    Parameters
    ----------
    dem : int or float
        Digital elevation model.
    fdr : int
        Flow direction.
    downslope : float array
        Downslope index array.
    px : int or float
        Raster pixel dimension.
    dif : int
        Elevation difference.
    row : int
        Number of rows of the 2-D matrix.
    col : int
        Number of rows of the 2-D matrix.

    '''
    i = cuda.grid(1)
    if i >= 0 and i <row*col:
        if dem[i] <= -100:
        # if dem[i] != -100:
            downslope[i] = -100
        else:
            pos = i
            isnan = 0
            dist = 0
            loop = 0
            out = 0
            while(dem[i] - dem[pos] < dif):
                if pos < col and (fdr[pos] == 32 or fdr[pos] == 64 or fdr[pos] == 128):
                    out = 1
                    break
                elif pos%col == 0 and (fdr[pos] == 8 or fdr[pos] == 16 or fdr[pos] == 32):
                    out = 1
                    break
                elif pos%col == (col-1) and (fdr[pos] == 128 or fdr[pos] == 1 or fdr[pos] == 2):
                    out = 1
                    break
                elif pos >= (row-1) * row and (fdr[pos] == 2 or fdr[pos] == 4 or fdr[pos] == 8):
                    out = 1
                    break
                
                if fdr[pos] == 1:
                    pos += 1
                    dist += px
                elif fdr[pos] == 2:
                    pos += 1 + col
                    dist += (px * math.sqrt(2.0))
                elif fdr[pos] == 4:
                    pos += col
                    dist += px
                elif fdr[pos] == 8:
                    pos += col -1
                    dist += (px * math.sqrt(2.0))
                elif fdr[pos] == 16:
                    pos += -1
                    dist += px
                elif fdr[pos] == 32:
                    pos += -1 - col
                    dist += (px * math.sqrt(2.0))
                elif fdr[pos] == 64:
                    pos += -col
                    dist += px
                elif fdr[pos] == 128:
                    pos += -col + 1
                    dist += (px * math.sqrt(2.0))
                if fdr[pos] == -100:
                    isnan = 1
                    break
                
                loop += 1
                if loop == 5000:
                    isnan = 1
                    break            
                
                #não tem que ter esse aqui?
                if dem[pos] == -100:
                    isnan = 1
                    break
            if isnan == 1:
                downslope[i] = -50
            elif out == 1:# or out ==2 or out ==3 or out ==4:
                downslope[i] = -50
                
            else:
                downslope[i] = (dem[i] - dem[pos]) / dist