from numba import cuda, jit, float32
import numpy as np
import math

import time


def divisor(row_length, column_length, row_division, column_division):
    bCol = np.array([], dtype=int)
    bRow = np.array([], dtype=int)

    for i in range(0, row_division, 1):
        bRow = np.append(
            bRow, [math.floor((i + 1) * row_length / (row_division + 1))])
    for i in range(0, column_division, 1):
        bCol = np.append(
            bCol,
            [math.floor((i + 1) * column_length / (column_division + 1))])

    return bRow, bCol


def downslope_sequential(dem,
                         flow_direction,
                         px,
                         elevation_difference,
                         downslope=np.array([])):
    '''
    Downslope sequential method. Also responsible for fixing cells that could
    not be simulated in the gpu.

    Parameters
    ----------
    dem : int or float
        Digital elevation model.
    flow_direction : int
        Flow direction.
    px : int or float
        Raster pixel dimension.
    elevation_difference : int
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
        downslope = np.zeros([len(dem), len(dem[0])], dtype='float32')
        new = 1

    for i in range(622, len(downslope), 1):
        for j in range(2020, len(downslope[0]), 1):
            if dem[i, j] == -100:
                downslope[i, j] = -100
                continue
            elif new == 0 and downslope[i, j] != -50:
                continue
            else:
                y = i
                x = j
                dist = 0
                loop = 0
                is_nan = 0
                while (dem[i, j] - dem[y, x] < elevation_difference):
                    if y == 0 and (flow_direction[y, x] == 32
                                   or flow_direction[y, x] == 64
                                   or flow_direction[y, x] == 128):
                        is_nan = 1
                        break
                    elif y == len(downslope) - 1 and (
                            flow_direction[y, x] == 2 or flow_direction[y, x]
                            == 4 or flow_direction[y, x] == 8):
                        is_nan = 1
                        break
                    elif x == 0 and (flow_direction[y, x] == 32
                                     or flow_direction[y, x] == 16
                                     or flow_direction[y, x] == 8):
                        is_nan = 1
                        break
                    elif x == len(downslope[0]) - 1 and (
                            flow_direction[y, x] == 128 or flow_direction[y, x]
                            == 1 or flow_direction[y, x] == 2):
                        is_nan = 1
                        break

                    if flow_direction[y, x] == 1:
                        if dem[y, x + 1] == -100:
                            is_nan = 2
                            break
                        x += 1
                        dist += px
                    elif flow_direction[y, x] == 2:
                        if dem[y + 1, x + 1] == -100:
                            is_nan = 3
                            break
                        x += 1
                        y += 1
                        dist += px * math.sqrt(2.0)  #Arrumar raiz aqui
                    elif flow_direction[y, x] == 4:
                        if dem[y + 1, x] == -100:
                            is_nan = 4
                            break
                        y += 1
                        dist += px
                    elif flow_direction[y, x] == 8:
                        if dem[y + 1, x - 1] == -100:
                            is_nan = 5
                            break
                        x -= 1
                        y += 1
                        dist += px * math.sqrt(2.0)  # FIX: Square root
                    elif flow_direction[y, x] == 16:
                        if dem[y, x - 1] == -100:
                            is_nan = 6
                            break
                        x -= 1
                        dist += px
                    elif flow_direction[y, x] == 32:
                        if dem[y - 1, x - 1] == -100:
                            is_nan = 7
                            break
                        x -= 1
                        y -= 1
                        dist += px * math.sqrt(2.0)  # FIX: Square root
                    elif flow_direction[y, x] == 64:
                        if dem[y - 1, x] == -100:
                            is_nan = 8
                            break
                        y -= 1
                        dist += px
                    elif flow_direction[y, x] == 128:
                        if dem[y - 1, x + 1] == -100:
                            is_nan = 9
                            break
                        x += 1
                        y -= 1
                        dist += px * math.sqrt(2.0)  # FIX: Square root
                    elif flow_direction[y, x] == -100:
                        is_nan = 10
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

                    if dem[y, x] == -100:
                        is_nan = 11
                        break

                    loop += 1

                    if loop == 500:
                        break

                if is_nan > 1:
                    downslope[i, j] = -100
                else:
                    downslope[i, j] = (dem[i, j] - dem[y, x]) / dist

    return downslope


@jit
def downslope_sequential_jit(dem,
                             flow_direction,
                             px,
                             elevation_difference,
                             downslope=np.array([[], []], 'float32')):
    '''
    Downslope sequential method. Also responsible for fixing cells that could
    not be simulated in the gpu.

    Parameters
    ----------
    dem : int or float
        Digital elevation model.
    flow_direction : int
        Flow direction.
    px : int or float
        Raster pixel dimension.
    elevation_difference : int
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
        downslope = np.zeros(dem.shape, dtype=float32)
        new = 1

    for i in range(0, len(downslope), 1):
        for j in range(0, len(downslope[0]), 1):
            if dem[i, j] == -100:
                downslope[i, j] = -100
                continue
            elif new == 0 and downslope[i, j] != -50:
                continue
            else:
                y = i
                x = j
                dist = 0
                loop = 0
                is_nan = 0
                while (dem[i, j] - dem[y, x] < elevation_difference):
                    if y == 0 and (flow_direction[y, x] == 32
                                   or flow_direction[y, x] == 64
                                   or flow_direction[y, x] == 128):
                        is_nan = 1
                        break
                    elif y == len(downslope) - 1 and (
                            flow_direction[y, x] == 2 or flow_direction[y, x]
                            == 4 or flow_direction[y, x] == 8):
                        is_nan = 1
                        break
                    elif x == 0 and (flow_direction[y, x] == 32
                                     or flow_direction[y, x] == 16
                                     or flow_direction[y, x] == 8):
                        is_nan = 1
                        break
                    elif x == len(downslope[0]) - 1 and (
                            flow_direction[y, x] == 128 or flow_direction[y, x]
                            == 1 or flow_direction[y, x] == 2):
                        is_nan = 1
                        break

                    if flow_direction[y, x] == 1:
                        if dem[y, x + 1] == -100:
                            is_nan = 1
                            break
                        x += 1
                        dist += px
                    elif flow_direction[y, x] == 2:
                        if dem[y + 1, x + 1] == -100:
                            is_nan = 1
                            break
                        x += 1
                        y += 1
                        dist += px * math.sqrt(2.0)  # FIX: Square root
                    elif flow_direction[y, x] == 4:
                        if dem[y + 1, x] == -100:
                            is_nan = 1
                            break
                        y += 1
                        dist += px
                    elif flow_direction[y, x] == 8:
                        if dem[y + 1, x - 1] == -100:
                            is_nan = 1
                            break
                        x -= 1
                        y += 1
                        dist += px * math.sqrt(2.0)  # FIX: Square root
                    elif flow_direction[y, x] == 16:
                        if dem[y, x - 1] == -100:
                            is_nan = 1
                            break
                        x -= 1
                        dist += px
                    elif flow_direction[y, x] == 32:
                        if dem[y - 1, x - 1] == -100:
                            is_nan = 1
                            break
                        x -= 1
                        y -= 1
                        dist += px * math.sqrt(2.0)  # FIX: Square root
                    elif flow_direction[y, x] == 64:
                        if dem[y - 1, x] == -100:
                            is_nan = 1
                            break
                        y -= 1
                        dist += px
                    elif flow_direction[y, x] == 128:
                        if dem[y - 1, x + 1] == -100:
                            is_nan = 1
                            break
                        x += 1
                        y -= 1
                        dist += px * math.sqrt(2.0)  # FIX: Square root
                    elif flow_direction[y, x] == -100:
                        is_nan = 1
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

                    if dem[y, x] == -100:
                        is_nan = 1
                        break

                    loop += 1

                    if loop == 5000:
                        break

                if is_nan == 1:
                    if dist == 0:
                        downslope[i, j] = 0
                    else:
                        downslope[i, j] = (dem[i, j] - dem[y, x]) / dist
                else:
                    downslope[i, j] = (dem[i, j] - dem[y, x]) / dist

    return downslope


def downsloper(dem,
               flow_direction,
               px,
               elevation_difference,
               column_division=0,
               row_division=0):
    '''
    Method responsible for the partioning of the matrix for the downslope.

    Parameters
    ----------
    dem : int or float
        Digital elevation model.
    flow_direction : int
        Flow direction.
    px : int or float
        Raster pixel dimension.
    elevation_difference : int
        Elevation difference.
    column_division : int, optional
        Number of vertical divisions. The default is 0.
    row_division : int, optional
        Number of horizontal divisions. The default is 0.

    Returns
    -------
    downslope : float array
        Downslope index array.

    '''
    row_size = len(dem)
    col_size = len(dem[0])

    bRow, bCol = divisor(row_size, col_size, row_division, column_division)

    downslope = np.zeros((row_size, col_size), dtype='float32')

    bRow = np.insert(bRow, row_division, row_size)
    bRow = np.insert(bRow, 0, 0)
    bCol = np.insert(bCol, column_division, col_size)
    bCol = np.insert(bCol, 0, 0)

    for m in range(0, row_division + 1, 1):
        for n in range(0, column_division + 1, 1):
            mS = bRow[m]
            mE = bRow[m + 1]
            nS = bCol[n]
            nE = bCol[n + 1]

            downslope[mS:mE,
                      nS:nE] = downslope_host(dem[mS:mE, nS:nE],
                                              flow_direction[mS:mE, nS:nE], px,
                                              elevation_difference)
    downslope = downslope_sequential_jit(dem, flow_direction, px,
                                         elevation_difference, downslope)

    return downslope


def downslope_host(dem,
                   flow_direction,
                   px,
                   elevation_difference,
                   blocks=0,
                   threads=0):
    '''
    Method responsible for the host/device data transfer

    Parameters
    ----------
    dem : int or float
        Digital elevation model.
    flow_direction : int
        Flow direction.
    px : int or float
        Raster pixel dimension.
    elevation_difference : int
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
        blocks = math.ceil((row * col) / threads)
        # FIX: this!!!!
    dem = np.asarray(dem).reshape(-1)
    flow_direction = np.asarray(flow_direction).reshape(-1)
    downslope = np.zeros((row * col))

    dem = cuda.to_device(dem)
    flow_direction = cuda.to_device(flow_direction)
    downslope = cuda.to_device(downslope)

    downslope_device[blocks, threads](dem, flow_direction, downslope, px,
                                      elevation_difference, col, row)

    downslope = downslope.copy_to_host()

    downslope = downslope.reshape(row, col)

    return downslope


@cuda.jit
def downslope_device(dem, flow_direction, downslope, px, elevation_difference,
                     col, row):
    '''
    GPU Downslope index method

    Parameters
    ----------
    dem : int or float
        Digital elevation model.
    flow_direction : int
        Flow direction.
    downslope : float array
        Downslope index array.
    px : int or float
        Raster pixel dimension.
    elevation_difference : int
        Elevation difference.
    row : int
        Number of rows of the 2-D matrix.
    col : int
        Number of rows of the 2-D matrix.

    '''
    i = cuda.grid(1)
    if i >= 0 and i < row * col:
        if dem[i] <= -100:
            downslope[i] = -100
        else:
            pos = i
            is_nan = 0
            dist = 0
            loop = 0
            out = 0
            while (dem[i] - dem[pos] < elevation_difference):
                if pos < col and (flow_direction[pos] == 32
                                  or flow_direction[pos] == 64
                                  or flow_direction[pos] == 128):
                    out = 1
                    break
                elif pos % col == 0 and (flow_direction[pos] == 8
                                         or flow_direction[pos] == 16
                                         or flow_direction[pos] == 32):
                    out = 1
                    break
                elif pos % col == (col - 1) and (flow_direction[pos] == 128
                                                 or flow_direction[pos] == 1
                                                 or flow_direction[pos] == 2):
                    out = 1
                    break
                elif pos >= (row - 1) * row and (flow_direction[pos] == 2
                                                 or flow_direction[pos] == 4
                                                 or flow_direction[pos] == 8):
                    out = 1
                    break

                if flow_direction[pos] == 1:
                    pos += 1
                    dist += px
                elif flow_direction[pos] == 2:
                    pos += 1 + col
                    dist += (px * math.sqrt(2.0))
                elif flow_direction[pos] == 4:
                    pos += col
                    dist += px
                elif flow_direction[pos] == 8:
                    pos += col - 1
                    dist += (px * math.sqrt(2.0))
                elif flow_direction[pos] == 16:
                    pos += -1
                    dist += px
                elif flow_direction[pos] == 32:
                    pos += -1 - col
                    dist += (px * math.sqrt(2.0))
                elif flow_direction[pos] == 64:
                    pos += -col
                    dist += px
                elif flow_direction[pos] == 128:
                    pos += -col + 1
                    dist += (px * math.sqrt(2.0))
                if flow_direction[pos] == -100:
                    is_nan = 1
                    break

                loop += 1
                if loop == 5000:
                    is_nan = 1
                    break

                if dem[pos] == -100:
                    is_nan = 1
                    break
            if is_nan == 1:
                downslope[i] = -50
            elif out == 1:
                downslope[i] = -50

            else:
                downslope[i] = (dem[i] - dem[pos]) / dist