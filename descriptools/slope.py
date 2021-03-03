from numba import cuda, jit
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


@jit
def slope_sequential_jit(dem, px):
    '''
    Return the highest slope to a neighbouring cell. Sequential implementation
    
    Parameters:
        dem : Digital evelation model
        px : Raster pixel dimensions
    '''
    row, col = dem.shape
    slope = np.zeros((row, col))
    for i in range(0, row, 1):
        for j in range(0, col, 1):
            aux = 0

            if dem[i, j] == -100:
                slope[i, j] = -100
                continue

            for y in range(-1, 2, 1):
                for x in range(-1, 2, 1):
                    if i + y < 0 or i + y >= row or j + x < 0 or j + x >= col:
                        continue
                    if x == 0 and y == 0:
                        continue
                    if dem[i + y, j + x] == -100:
                        continue

                    if x == 0 or y == 0:
                        if aux < (dem[i, j] - dem[i + y, j + x]) / px:
                            aux = (dem[i, j] - dem[i + y, j + x]) / px
                        else:
                            continue
                    else:
                        if aux < (dem[i, j] - dem[i + y, j + x]) / (px *
                                                                    1.4142):
                            aux = (dem[i, j] - dem[i + y, j + x]) / (px *
                                                                     1.4142)
                        else:
                            continue
            slope[i, j] = aux * 100
    return slope


def slope_sequential(dem, px):
    '''
    Return the highest slope to a neighbouring cell. Sequential implementation
    
    Parameters:
        dem : Digital evelation model
        px : Raster pixel dimensions
    '''
    row, col = dem.shape
    slope = np.zeros((row, col))
    for i in range(2, row, 1):
        for j in range(190, col, 1):
            aux = 0

            if dem[i, j] == -100:
                slope[i, j] = -100
                continue

            for y in range(-1, 2, 1):
                for x in range(-1, 2, 1):
                    if i + y < 0 or i + y >= row or j + x < 0 or j + x >= col:
                        continue
                    if x == 0 and y == 0:
                        continue
                    if dem[i + y, j + x] == -100:
                        continue

                    if x == 0 or y == 0:
                        if aux < (dem[i, j] - dem[i + y, j + x]) / px:
                            aux = (dem[i, j] - dem[i + y, j + x]) / px
                        else:
                            continue
                    else:
                        # var = (dem[i,j] - dem[i+y,j+x])/(px*1.4142)
                        if aux < (dem[i, j] - dem[i + y, j + x]) / (px *
                                                                    1.4142):
                            aux = (dem[i, j] - dem[i + y, j + x]) / (px *
                                                                     1.4142)
                        else:
                            continue
            slope[i, j] = aux
    return slope


def sloper(dem, px):
    '''
    Method responsible for the matrix dimension division

    Parameters
    ----------
    dem : int or float
        Digital elevation model.
    px : int or float
        Raster pixel dimension.

    Returns
    -------
    slope : float
        Highest slope of the neighbouring cells.

    '''
    #Essa é a função responsável pela divisão
    #Slope é simples, divide em x partes e adiciona colunas dos lados
    #Se pegar coluna/linha de fora, adiciona como nan, se não pega da outra divisão

    div_col = 0
    div_row = 0

    row_size = len(dem)
    col_size = len(dem[0])

    bRow, bCol = divisor(row_size, col_size, div_row, div_col)

    slope = np.zeros((row_size, col_size))

    bRow = np.insert(bRow, div_row, row_size)
    bRow = np.insert(bRow, 0, 0)
    bCol = np.insert(bCol, div_col, col_size)
    bCol = np.insert(bCol, 0, 0)

    for m in range(0, div_row + 1, 1):
        for n in range(0, div_col + 1, +1):

            mS = bRow[m]
            mE = bRow[m + 1]
            nS = bCol[n]
            nE = bCol[n + 1]

            # print(dem[mS:mE,nS:nE])
            extra = np.array([0, 0, 0, 0])

            if m == 0:
                extra[0] = 1  #cima
            if n == 0:
                extra[1] = 1  #esquerda
            if n == div_col:
                extra[2] = 1  #direita
            if m == div_row:
                extra[3] = 1  #baixo

            slope[mS:mE, nS:nE] = slope_host(
                dem[mS - 1 + extra[0]:mE + 1 - extra[3],
                    nS - 1 + extra[1]:nE + 1 - extra[2]], px, extra)

    return slope


def slope_host(dem, px, extra, blocks=0, threads=0):
    '''
    Method responsible for the host-device data transfer and method calling.

    Parameters
    ----------
    dem : int or float
        digital elevation model.
    px : int or float
        raster pixel dimension.
    extra : int or float
        Extra columns and rows necessary for the first/last column/row cells.
    blocks : int, optional
        Number of block of threads. The default is 0.
    threads : int, optional
        number of threads in each block. The default is 0.

    Returns
    -------
    slope : int or float
        DESCRIPTION.

    '''
    if extra[0] == 1:
        dem = np.insert(dem, 0, -100, axis=0)
    if extra[1] == 1:
        dem = np.insert(dem, 0, -100, axis=1)
    if extra[2] == 1:
        dem = np.insert(dem, len(dem[0]), -100, axis=1)
    if extra[3] == 1:
        dem = np.insert(dem, len(dem), -100, axis=0)

    row = len(dem)
    col = len(dem[0])

    if blocks == 0 and threads == 0:
        threads = 256
        blocks = math.ceil((row * col) / threads)
        #checar isso aqui

    dem = np.asarray(dem).reshape(-1)
    slope = np.zeros((row * col), dtype='float32')

    dem = cuda.to_device(dem)
    slope = cuda.to_device(slope)

    slope_device[blocks, threads](dem, slope, px, row, col)

    slope = slope.copy_to_host()
    slope = slope.reshape(row, col)
    slope = np.delete(slope, 0, axis=0)
    slope = np.delete(slope, 0, axis=1)
    slope = np.delete(slope, len(slope) - 1, axis=0)
    slope = np.delete(slope, len(slope[0]) - 1, axis=1)
    return slope


@cuda.jit
def slope_device(dem, slope, px, row, col):
    '''
    GPU slope method

    Parameters
    ----------
    dem : int or float
        Digital elevation model.
    slope : float
        Slope matrix that will be filled and returned. Initialized as zeros.
    px : int or float
        Rasters pixel dimension size.
    row : int
        Number of rows of the 2-D matrix.
    col : int
        Number of rows of the 2-D matrix.

    '''
    i = cuda.grid(1)

    if i >= 0 and i < col * row:
        if dem[i] <= -100:
            slope[i] = -100
        else:
            #essas condições impedem que calculem na linhas e colunas extras
            if i % col == 0:  ## Primeira coluna
                slope[i] = -100
            elif i < col:
                slope[i] = -100  ## primeira linha
            elif i % col == col - 1:
                slope[i] = -100  ## última coluna
            elif i >= col * (row - 1):
                slope[i] = -100  ## última linha
            else:
                aux = 0
                for y in range(-1, 2, 1):
                    for x in range(-1, 2, 1):
                        pos = i + (y * col) + x
                        if dem[pos] == -100:
                            continue
                        if x == 0 or y == 0:
                            if aux < (dem[i] - dem[pos]) / px:
                                aux = (dem[i] - dem[pos]) / px
                            else:
                                continue
                        else:
                            if aux < (dem[i] - dem[pos]) / (px * 1.4142):
                                aux = (dem[i] - dem[pos]) / (px * 1.4142)
                            else:
                                continue
            slope[i] = aux * 100


#
# dem = np.array([[16,15,14,14,16,17,19,15,14,15,17],
#       [14,13, 12, 13, 14, 15, 15, 14, 12, 13, 15],
#       [15,12, 11, 12, 14, 13, 12, 11, 13, 14, 14],
#       [14,12, 10, 13, 12, 11, 10, 14, 13, 12, 11],
#       [15,13, 11, 9, 11, 10, 9, 12, 8, 12, 9],
#       [16,15, 13, 11, 8, 8, 9, 10, 7, 7, 6],
#       [14,12, 14, 8, 9, 7, 8, 7, 3, 4, 3],
#       [12,11, 9, 11, 7, 6, 8, 6, 6, 2, 1],
#       [12,10, 11, 9, 7, 7, 5, 4, 3, 4, 2],
#       [13,12, 11, 10, 8, 7, 7, 5, 5, 7, 6],
#       [15,14, 12, 12, 11, 10, 9, 7, 8, 9, 9]], dtype='float32')

# px = 1

# slope = slope(dem, px)

# extra = np.array([1,1,1,1])

# if extra[0] == 1:
#     dem = np.insert(dem,0,-100,axis = 0)
# if extra[1] == 1:
#     dem = np.insert(dem,0,-100,axis = 1)
# if extra[2] == 1:
#     dem = np.insert(dem,len(dem[0]),-100,axis = 1)
# if extra[3] == 1:
#     dem = np.insert(dem,len(dem),-100,axis = 0)

# row = len(dem)
# col = len(dem[0])

# dem = np.asarray(dem).reshape(-1)
# slope2 = np.zeros((row*col), dtype='float32')

# for i in range(0,len(dem),1):
#     if dem[i] <= -100:
#         slope2[i] = -100
#     else:
#         #essas condições impedem que calculem na linhas e colunas extras
#         if i%col == 0: ## Primeira coluna
#             slope2[i] = -100
#         elif i < col:
#             slope2[i] = -100 ## primeira linha
#         elif i%col== col-1:
#             slope2[i] = -100 ## última coluna
#         elif i >= col*(row-1):
#             slope2[i] = -100 ## última linha
#         else:
#             aux = 0;
#             for y in range(-1,2,1):
#                 for x in range(-1,2,1):
#                     pos = i + (y*col) + x
#                     if dem[pos] == -100:
#                         continue
#                     if x == 0 or y == 0:
#                         if aux < (dem[i] - dem[pos])/px:
#                             aux = (dem[i] - dem[pos])/px
#                         else:
#                             continue
#                     else:
#                         if aux < (dem[i] - dem[pos])/(px*1.4142):
#                             aux = (dem[i] - dem[pos])/(px*1.4142)
#                         else:
#                             continue

#         slope2[i] = aux * 100

####
# def plotFigure(data, label, cmap='Blues'):
#     plt.figure(figsize=(12,10))
#     plt.imshow(data,extent=grid.extent)
#     plt.colorbar(label=label)
#     # plt.set_cmap(cmap)
#     plt.grid()
#     plt.show()
