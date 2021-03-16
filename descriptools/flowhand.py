from numba import cuda, jit, int32, float32
import numpy as np
import math

from descriptools.helpers import divisor


def flow_distance_indexes_sequential(flow_direction,
                                     river_matrix,
                                     px,
                                     flow_distance=np.array([])):
    '''
    Flow distance to the nearest drainage sequential method. Also preemptively 
    calculates the row and columns required for the matrix division.

    Parameters
    ----------
    flow_direction : int8 array
        Flow direction.
    river_matrix : int8 array
        Binary matrix that 1- river cell 0-not river cell.
    px : int or float
        Raster pixel dimension.
    flow_distance : flow distance float array, optional
        Flow distance. The default is an empty array.
        This input is used for the boundary array calculation for matrix subdivision.

    Returns
    -------
    flow_distance : float array
        Flow distance.
    indices : int array
        River cell index.

    '''
    new = 0
    if flow_distance.size == 0:
        flow_distance = np.zeros(
            [len(flow_direction), len(flow_direction[0])], dtype='float32')
        new = 1

    indices = np.zeros(
        [len(flow_direction), len(flow_direction[0])], dtype='int')
    for i in range(0, len(flow_direction), 1):
        for j in range(0, len(flow_direction[0]), 1):
            if flow_direction[i, j] == 0:
                flow_distance[i, j] = -100
                indices[i, j] = -100
            elif new == 0 and flow_distance[i, j] != -50:
                continue
            else:
                y = i
                x = j
                dist = 0
                loop = 0
                isnan = 0
                while (river_matrix[y, x] != 1):
                    if y == 0 and (flow_direction[y, x] == 32
                                   or flow_direction[y, x] == 64
                                   or flow_direction[y, x] == 128):
                        isnan = 1
                        break
                    elif y == len(flow_distance) - 1 and (
                            flow_direction[y, x] == 2 or flow_direction[y, x]
                            == 4 or flow_direction[y, x] == 8):
                        isnan = 1
                        break
                    elif x == 0 and (flow_direction[y, x] == 32
                                     or flow_direction[y, x] == 16
                                     or flow_direction[y, x] == 8):
                        isnan = 1
                        break
                    elif x == len(flow_distance[0]) - 1 and (
                            flow_direction[y, x] == 128 or flow_direction[y, x]
                            == 1 or flow_direction[y, x] == 2):
                        isnan = 1
                        break

                    if flow_direction[y, x] == 1:
                        x += 1
                        dist += px
                    elif flow_direction[y, x] == 2:
                        x += 1
                        y += 1
                        dist += px * math.sqrt(2.0)  # FIX: Square root
                    elif flow_direction[y, x] == 4:
                        y += 1
                        dist += px
                    elif flow_direction[y, x] == 8:
                        x -= 1
                        y += 1
                        dist += px * math.sqrt(2.0)  # FIX: Square root
                    elif flow_direction[y, x] == 16:
                        x -= 1
                        dist += px
                    elif flow_direction[y, x] == 32:
                        x -= 1
                        y -= 1
                        dist += px * math.sqrt(2.0) # FIX: Square root
                    elif flow_direction[y, x] == 64:
                        y -= 1
                        dist += px
                    elif flow_direction[y, x] == 128:
                        x += 1
                        y -= 1
                        dist += px * math.sqrt(2.0)  # FIX: Square root

                    elif flow_direction[y, x] == 255:
                        isnan = 1
                        break

                    loop += 1
                    if loop >= 5000:
                        isnan = 1
                        break

                if isnan == 1:
                    indices[i, j] = -100
                    flow_distance[i, j] = -100
                else:
                    flow_distance[i, j] = dist
                    indices[i, j] = y * len(flow_direction[0]) + x

    return flow_distance, indices


@jit
def fdist_indexes_sequential_jit(fdr,
                                 river,
                                 px,
                                 fdist=np.array([[], []], 'float32')):
    '''
    Flow distance to the nearest drainage sequential method. Also preemptively 
    calculates the row and columns required for the matrix division.

    Parameters
    ----------
    fdr : int8 array
        Flow direction.
    river : int8 array
        Binary matrix that 1- river cell 0-not river cell.
    px : int or float
        Raster pixel dimension.
    fdist : flow distance float array, optional
        Flow distance. The default is an empty array.
        This input is used for the boundary array calculation for matrix subdivision.

    Returns
    -------
    fdist : float array
        Flow distance.
    indices : int array
        River cell index.

    '''
    new = 0
    if fdist.size == 0:
        fdist = np.zeros(fdr.shape, dtype=float32)
        new = 1

    indices = np.zeros(fdr.shape, int32)
    for i in range(0, len(fdr), 1):
        for j in range(0, len(fdr[0]), 1):
            if new == 0 and fdist[i, j] != -50:
                continue
            elif fdr[i, j] == 0:
                fdist[i, j] = -100
                indices[i, j] = -100
            else:
                y = i
                x = j
                dist = 0
                loop = 0
                isnan = 0
                while (river[y, x] != 1):
                    if y == 0 and (fdr[y, x] == 32 or fdr[y, x] == 64
                                   or fdr[y, x] == 128):
                        isnan = 1
                        break
                    elif y == len(fdist) - 1 and (fdr[y, x] == 2 or fdr[y, x]
                                                  == 4 or fdr[y, x] == 8):
                        isnan = 1
                        break
                    elif x == 0 and (fdr[y, x] == 32 or fdr[y, x] == 16
                                     or fdr[y, x] == 8):
                        isnan = 1
                        break
                    elif x == len(fdist[0]) - 1 and (fdr[y, x] == 128
                                                     or fdr[y, x] == 1
                                                     or fdr[y, x] == 2):
                        isnan = 1
                        break

                    if fdr[y, x] == 1:
                        x += 1
                        dist += px
                    elif fdr[y, x] == 2:
                        x += 1
                        y += 1
                        dist += px * math.sqrt(2.0)  # FIX: Square root
                    elif fdr[y, x] == 4:
                        y += 1
                        dist += px
                    elif fdr[y, x] == 8:
                        x -= 1
                        y += 1
                        dist += px * math.sqrt(2.0)  # FIX: Square root
                    elif fdr[y, x] == 16:
                        x -= 1
                        dist += px
                    elif fdr[y, x] == 32:
                        x -= 1
                        y -= 1
                        dist += px * math.sqrt(2.0)  # FIX: Square root
                    elif fdr[y, x] == 64:
                        y -= 1
                        dist += px
                    elif fdr[y, x] == 128:
                        x += 1
                        y -= 1
                        dist += px * math.sqrt(2.0)  # FIX: Square root

                    elif fdr[y, x] == 0:
                        isnan = 1
                        break

                    loop += 1
                    if loop >= 5000:
                        isnan = 1
                        break

                if isnan == 1:
                    indices[i, j] = -100
                    fdist[i, j] = -100
                else:
                    fdist[i, j] = dist
                    indices[i, j] = y * len(fdr[0]) + x

    return fdist, indices


def flow_hand_index(dem_raster,
                    flow_direction_matrix,
                    river_matrix,
                    px,
                    division_column=0,
                    division_row=0):
    '''
    Overall method responsible for the matrix partitioning and
    defining dimensions.

    Parameters
    ----------
    dem_raster : int or float array
        Digital elevation model raster.
    flow_direction_matrix : int8 array
        Flow direction.
    river_matrix : int8 array
        Binary matrix that 1- river cell 0-not river cell.
    px : int or float
        Raster pixel dimension.

    Returns
    -------
    fdist : float array
        Flow distance.
    indices : int array
        River cell index.
    hand : int or flat array
        Height above the nearest drainage.

    '''
    row = len(dem_raster)
    col = len(dem_raster[0])

    boundary_row, boundary_column = divisor(row, col, division_row,
                                            division_column)

    flow_distance = np.zeros((row, col),dtype='float32')
    indices = np.zeros((row, col), dtype='int')

    if division_row > 0 or division_column > 0:
        flow_distance[:, boundary_column] = -50
        flow_distance[boundary_row, :] = -50
        flow_distance, indices = fdist_indexes_sequential_jit(
            flow_direction_matrix, river_matrix, px, flow_distance)

    boundary_row = np.insert(boundary_row, division_row, row)
    boundary_row = np.insert(boundary_row, 0, -1)
    boundary_column = np.insert(boundary_column, division_column, col)
    boundary_column = np.insert(boundary_column, 0, -1)

    part = 0

    for m in range(0, division_row + 1, 1):
        for n in range(0, division_column + 1, 1):
            out = np.zeros((4))
            bound_c = np.zeros([1])
            bound_e = np.zeros([1])
            bound_d = np.zeros([1])
            bound_b = np.zeros([1])
            
            index_up    = np.zeros([1])
            index_left  = np.zeros([1])
            index_right = np.zeros([1])
            index_down  = np.zeros([1])

            mS = boundary_row[m]
            mE = boundary_row[m + 1]
            nS = boundary_column[n]
            nE = boundary_column[n + 1]

            if division_row > 0:
                if part < (division_column + 1) * division_row:
                    out[3] = 1
                    bound_b = flow_distance[mE, nS + 1:nE]
                    index_down = indices[mE, nS + 1:nE]
                    if n != division_column:
                        bound_b = np.insert(bound_b, nE - (nS + 1),
                                            flow_distance[mE, nE])
                        index_down = np.insert(index_down, nE - (nS + 1),
                                            indices[mE, nE])
                        
                    if n != 0:
                        bound_b = np.insert(bound_b, 0, flow_distance[mE, nS])
                        index_down = np.insert(index_down, 0, indices[mE, nS])

                if part > division_column:
                    out[0] = 1
                        
                    bound_c = flow_distance[mS, nS + 1:nE]
                    index_up = indices[mS, nS + 1:nE]
                    
                    if n != division_column:
                        bound_c = np.insert(bound_c, nE - (nS + 1),
                                            flow_distance[mS, nE])
                        index_up = np.insert(index_up, nE - (nS + 1),
                                            indices[mS, nE])
                    if n != 0:
                        bound_c = np.insert(bound_c, 0, flow_distance[mS, nS])
                        index_up = np.insert(index_up, 0, indices[mS, nS])

            if division_column > 0:
                if part % (division_column + 1) != 0:
                    out[1] = 1
                    bound_e = flow_distance[mS + 1:mE, nS]
                    index_left = indices[mS + 1:mE, nS]
                    
                    if m != division_row:
                        bound_e = np.insert(bound_e, mE - (mS + 1),
                                            flow_distance[mE, nS])
                        index_left = np.insert(index_left, mE - (mS + 1),
                                            indices[mE, nS])
                        
                    if m != 0:
                        bound_e = np.insert(bound_e, 0, flow_distance[mS, nS])
                        index_left = np.insert(index_left, 0, indices[mS, nS])

                if part % (division_column + 1) != division_column:
                    out[2] = 1
                    bound_d = flow_distance[mS + 1:mE, nE]
                    index_right = indices[mS + 1:mE, nE]
                    
                    if m != division_row:
                        bound_d = np.insert(bound_d, mE - (mS + 1),
                                            flow_distance[mE, nE])
                        index_right = np.insert(index_right, mE - (mS + 1),
                                            indices[mE, nE])
                        
                    if m != 0:
                        bound_d = np.insert(bound_d, 0, flow_distance[mS, nE])
                        index_right = np.insert(index_right, 0, indices[mS, nE])
                        

            size = max(len(bound_c), len(bound_e), len(bound_d), len(bound_b))

            bound = np.zeros((4, size))
            bound_index = np.zeros((4, size))
            
            bound[0, 0:len(bound_c)] = bound_c
            bound_index[0, 0:len(index_up)] = index_up
            
            bound[1, 0:len(bound_e)] = bound_e
            bound_index[1, 0:len(index_left)] = index_left
            
            bound[2, 0:len(bound_d)] = bound_d
            bound_index[2, 0:len(index_right)] = index_right
            
            bound[3, 0:len(bound_b)] = bound_b
            bound_index[3, 0:len(index_down)] = index_down       

            mS += 1
            nS += 1
            part += 1
            
            flow_distance[mS:mE,
                          nS:nE], indices[mS:mE,
                                          nS:nE] = flow_distance_index_cpu(
                                              dem_raster[mS:mE, nS:nE],
                                              flow_direction_matrix[mS:mE,
                                                                    nS:nE],
                                              river_matrix[mS:mE, nS:nE], px,
                                              bound, bound_index, out, 
                                              mS, nS, col)

            # indices[mS:mE, nS:nE] = index_calculator(indices[mS:mE, nS:nE], mS,
            #                                           nS, col)

    hand = hand_calculator(dem_raster, indices)

    return flow_distance, indices, hand


def hand_calculator(dem, indices):
    '''
    Method that calculates the HAND index for each cell and its river cell

    Parameters
    ----------
    dem : int or float
        Digital elevation model.
    indices : int array
        River cell index.

    Returns
    -------
    hand : int or flat array
        Height above the nearest drainage.

    '''
    row, col = dem.shape
    dem = np.asarray(dem).reshape(-1)
    indices = np.asarray(indices).reshape(-1)
    hand = np.zeros(dem.shape)

    hand = np.where((dem != -100) & (indices != -100), dem - dem[indices],
                    -100)
    hand = np.where((hand < 0) & (hand != -100), 0, hand)

    hand = hand.reshape(row, col)

    return hand


def index_calculator(river_indices, row_start, column_start, column_size):
    '''
    Method that transform the river index from the sub-matrix to the whole matrix

    Parameters
    ----------
    river_indices : int array
        Sub matrix river cell index.
    row_start : int
        Position of the first row of the sub-matrix.
    column_start : int
        Position of the first column of the sub-matrix.
    column_size : int
        Number of columns in the whole matrix.

    Returns
    -------
    river_indices : int array
        River cell index.

    '''
    row, col = river_indices.shape

    river_indices = np.where(
        river_indices == -100, -100,
        (np.floor(river_indices / col) + row_start) * column_size +
        river_indices % col + column_start)

    return river_indices


def flow_distance_index_cpu(dem,
                            flow_direction,
                            river_matrix,
                            px,
                            boundary_distance,
                            boundary_index,
                            out,
                            row_start,
                            col_start,
                            matrix_columns,
                            blocks=0,
                            threads=0):
    '''
    Method responsible for the host/device data transfer

    Parameters
    ----------
    dem : int or float array
        Digital elevation model raster.
    flow_direction : int8 array
        Flow direction.
    river_matrix : int8 array
        Binary matrix that 1- river cell 0-not river cell.
    px : int or float
        Raster pixel dimension.
    boundary_distance : float array
        Bounduary array that contains values of flow distance. 
        Necessary for flow directions that lead to outside of the sub-matrix domain.
    boundary_index : int array
        Bounduary array that contains values of river indexes. 
        Necessary for flow directions that lead to outside of the sub-matrix domain.
    out : int array
        Array that represents wich directions have another sub-matrix.
        1-True 0-False. Up, left, right, down.
    row_start : int
        Row index of the first value
    col_start : int
        Column index of the first value
    matrix_columns : int
        Number of columns of the whole matrix
    blocks : int, optional
        Number of block of threads. The default is 0.
    threads : int, optional
        number of threads in each block. The default is 0.

    Returns
    -------
    flow_distance : float array
        Flow distance.
    indices : int array
        River cell index.

    '''
    row = len(dem)
    col = len(dem[0])

    if blocks == 0 and threads == 0:
        threads = 256
        blocks = math.ceil((row * col) / threads)

    dem = np.asarray(dem).reshape(-1)
    flow_direction = np.asarray(flow_direction).reshape(-1)
    river_matrix = np.asarray(river_matrix).reshape(-1)

    flow_distance = np.zeros((row * col), dtype='float32')
    indices = np.zeros((row * col))

    flow_direction = cuda.to_device(flow_direction)
    flow_distance = cuda.to_device(flow_distance)
    indices = cuda.to_device(indices)
    out = cuda.to_device(out)
    boundary_distance = cuda.to_device(boundary_distance)
    boundary_index = cuda.to_device(boundary_index)

    flow_distance_index_gpu[blocks,
                            threads](flow_direction, river_matrix, px,
                                     flow_distance, indices, col, row,
                                     boundary_distance, boundary_index, out,
                                     row_start, col_start, matrix_columns)

    indices = indices.copy_to_host()
    flow_distance = flow_distance.copy_to_host()

    indices = indices.reshape(row, col)
    flow_distance = flow_distance.reshape(row, col)

    return flow_distance, indices


@cuda.jit
def flow_distance_index_gpu(flow_direction, river_matrix, px, flow_distance,
                            indices, col, row, boundary_distance,
                            boundary_index, out, row_start, col_start, matrix_columns):
    '''
    GPU flow distance and river cell indexes method

    Parameters
    ----------
    flow_direction : int8 array
        Flow direction.
    river_matrix : int8 array
        Binary matrix that 1- river cell 0-not river cell.
    px : int or float
        Raster pixel dimension.
    flow_distance : float array
        Flow distance.
    indices : int array
        River cell index.
    row : int
        Number of rows of the 2-D matrix.
    col : int
        Number of rows of the 2-D matrix.
    boundary_distance : float array
        Bounduary array that contains values of flow distance. 
        Necessary for flow directions that lead to outside of the sub-matrix domain.
    boundary_index : int array
        Bounduary array that contains values of river indexes. 
        Necessary for flow directions that lead to outside of the sub-matrix domain.
    out : int array
        Array that represents wich directions have another sub-matrix.
        1-True 0-False. row0 Up, row1 left, row2 right, row3 down.

    '''
    i = cuda.grid(1)
    if i >= 0 and i < col * row:
        if flow_direction[i] <= 0:
            flow_distance[i] = -100
            indices[i] = -100
        
        # elif i == 1 or i == 2:
        #     flow_distance[i] = -33
        #     indices[i] = -33
        else:
            if river_matrix[i] == 1:
                flow_distance[i] = 0
                # indices[i] = i
                indices[i] = (row_start + math.floor(i/col)) * matrix_columns + col_start + i%col
            else:
                pos = i
                isnan = 0
                dist = 0
                loop = 0
                loop1 = -10
                loop2 = -20
                loop3 = -30
                isout = 0
                while (river_matrix[pos] != 1):
                    if pos < col and (flow_direction[pos] == 32
                                      or flow_direction[pos] == 64
                                      or flow_direction[pos] == 128):
                        if out[0] == 0:
                            isnan = 1
                            break
                        else:
                            initial = pos
                            irow = int(initial)
                            if out[1] == 1:
                                irow += 1
                            if flow_direction[pos] == 32:
                                if pos == 0 and out[1] == 0:
                                    isnan = 1
                                    break
                                else:
                                    irow = irow - 1
                                    if boundary_distance[0][irow] == -100:
                                        isnan = 1
                                        break
                                    dist += (px * math.sqrt(2.0)
                                             ) + boundary_distance[0][irow]
                                    indices[i] = boundary_index[0][irow]
                                    isout = 1
                                    break
                            elif flow_direction[pos] == 128:
                                if pos == col - 1 and out[2] == 0:
                                    isnan = 1
                                    break
                                else:
                                    irow = irow + 1
                                    if boundary_distance[0][irow] == -100:
                                        isnan = 1
                                        break
                                    dist += px * math.sqrt(2.0) + boundary_distance[0][
                                        irow]
                                    indices[i] = boundary_index[0][irow]
                                    isout = 1
                                    break
                            else:
                                if boundary_distance[0][irow] == -100:
                                        isnan = 1
                                        break
                                dist += px + boundary_distance[0][irow]
                                indices[i] = boundary_index[0][irow]
                                isout = 1
                                break

                    elif pos % col == 0 and (flow_direction[pos] == 8
                                             or flow_direction[pos] == 16
                                             or flow_direction[pos] == 32):

                        if out[1] == 0:
                            isnan = 1
                            break
                        else:
                            initial = pos / col
                            irow = int(initial)
                            if out[0] == 1:
                                irow += 1
                            if flow_direction[pos] == 32:
                                irow = irow - 1
                                if  boundary_distance[1][irow] == -100:
                                    isnan = 1
                                    break
                                dist += px * math.sqrt(2.0) + boundary_distance[1][irow]
                                indices[i] = boundary_index[1][irow]
                                isout = 1
                                break
                            elif flow_direction[pos] == 8:
                                if out[3] == 0 and pos == row * (col - 1):
                                    isnan = 1
                                    break
                                else:
                                    irow = irow + 1
                                    if  boundary_distance[1][irow] == -100:
                                        isnan = 1
                                        break
                                    dist += px * math.sqrt(2.0) + boundary_distance[1][
                                        irow]
                                    indices[i] = boundary_index[1][irow]
                                    isout = 1
                                    break
                            else:
                                if  boundary_distance[1][irow] == -100:
                                    isnan = 1
                                    break
                                dist += px + boundary_distance[1][irow]
                                indices[i] = boundary_index[1][irow]
                                isout = 1
                                break

                    elif pos % col == (col -
                                       1) and (flow_direction[pos] == 128
                                               or flow_direction[pos] == 1
                                               or flow_direction[pos] == 2):
                        if out[2] == 0:
                            isnan = 1
                            break
                        else:
                            initial = (pos + 1) / col - 1
                            irow = int(initial)
                            if out[0] == 1:
                                irow += 1
                            if flow_direction[pos] == 128:
                                irow = irow - 1
                                if  boundary_distance[2][irow] == -100:
                                    isnan = 1
                                    break
                                dist += px * math.sqrt(2.0) + boundary_distance[2][irow]
                                indices[i] = boundary_index[2][irow]
                                isout = 1
                                break
                            elif flow_direction[pos] == 2:
                                if out[3] == 0 and pos == (row * col) - 1:
                                    isnan = 1
                                    break
                                else:
                                    irow = irow + 1
                                    if  boundary_distance[2][irow] == -100:
                                        isnan = 1
                                        break
                                    dist += px * math.sqrt(2.0) + boundary_distance[2][
                                        irow]
                                    indices[i] = boundary_index[2][irow]
                                    isout = 1
                                    break
                            else:
                                if boundary_distance[2][irow] == -100:
                                    isnan = 1
                                    break
                                dist += px + boundary_distance[2][irow]
                                indices[i] = boundary_index[2][irow]
                                isout = 1
                                break

                    elif pos >= (row - 1) * col and (
                            flow_direction[pos] == 2 or flow_direction[pos]
                            == 4 or flow_direction[pos] == 8):
                        if out[3] == 0:
                            isnan = 1
                            break
                        else:
                            initial = pos - (row - 1) * col
                            irow = int(initial)
                            if out[1] == 1:
                                irow += 1
                            if flow_direction[pos] == 8:
                                irow = irow - 1
                                if  boundary_distance[3][irow] == -100:
                                    isnan = 1
                                    break
                                dist += px * math.sqrt(2.0) + boundary_distance[3][irow]
                                indices[i] = boundary_index[3][irow]
                                isout = 1
                                break
                            elif flow_direction[pos] == 2:
                                irow += 1
                                if  boundary_distance[3][irow] == -100:
                                    isnan = 1
                                    break
                                dist += px * math.sqrt(2.0) + boundary_distance[3][irow]
                                indices[i] = boundary_index[3][irow]
                                isout = 1
                                break
                            else:
                                if  boundary_distance[3][irow] == -100:
                                    isnan = 1
                                    break
                                dist += px + boundary_distance[3][irow]
                                indices[i] = boundary_index[3][irow]
                                isout = 1
                                break

                    loop3 = loop2
                    loop2 = loop1
                    loop1 = pos

                    if flow_direction[pos] == 1:
                        pos += 1
                        dist += px
                    elif flow_direction[pos] == 2:
                        pos += 1 + col
                        dist += px * math.sqrt(2.0)
                    elif flow_direction[pos] == 4:
                        pos += col
                        dist += px
                    elif flow_direction[pos] == 8:
                        pos += col - 1
                        dist += px * math.sqrt(2.0)
                    elif flow_direction[pos] == 16:
                        pos += -1
                        dist += px
                    elif flow_direction[pos] == 32:
                        pos += -1 - col
                        dist += px * math.sqrt(2.0)
                    elif flow_direction[pos] == 64:
                        pos += -col
                        dist += px
                    elif flow_direction[pos] == 128:
                        pos += -col + 1
                        dist += px * math.sqrt(2.0)

                    if flow_direction[pos] == 0:
                        isnan = 1
                        break

                    if pos == loop1 or pos == loop2 or pos == loop3:
                        isnan = 1
                        break

                    loop += 1
                    if loop > 20000:
                        isnan = 1
                        break

                if isnan == 1:
                    flow_distance[i] = -100
                    indices[i] = -100
                else:
                    flow_distance[i] = dist
                    if isout == 0:
                        indices[i] = (row_start + math.floor(pos/col)) * matrix_columns + col_start + pos%col
                    # indices[i] = pos
