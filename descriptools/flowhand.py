# Anotações:
# Problema conhecido: O 2**(1/2) da cpu é diferente da gpu, tem que padronizar isso
# Falta arrumar os índices e o CUDA
# Tipo, as variáveis estão erradas, ocupando muito mais memória que o necessário
# Ainda tem que resolver a questão de threads e blocos, já sei que no exemplo
# da bacia do rio itajai não funfa com só uma divisão
# na matriz de exemplo 11x11, o fdist tá dando uams com -100, tem que ver porque
#

from numba import cuda, jit, int32, float32
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


def fdist_indexes_sequential(fdr, river, px, fdist=np.array([])):
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
    fdist : flow distance flot array, optional
        Flow distance. The default is np.array([]).

    Returns
    -------
    fdist : float array
        Flow distance.
    indices : int array
        River cell index.

    '''
    new = 0
    if fdist.size == 0:
        fdist = np.zeros([len(fdr), len(fdr[0])], dtype='float32')
        new = 1

    indices = np.zeros([len(fdr), len(fdr[0])], dtype='int')
    for i in range(0, len(fdr), 1):
        for j in range(0, len(fdr[0]), 1):
            if fdr[i, j] == 255:
                fdist[i, j] = -100
                indices[i, j] = -100
            elif new == 0 and fdist[i, j] != -50:
                continue
            else:
                y = i
                x = j
                dist = 0
                loop = 0
                isnan = 0
                while (river[y, x] != 1):
                    #alguma condição de fronteira aqui
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
                        dist += px * 1.41  #Arrumar raiz aqui
                    elif fdr[y, x] == 4:
                        y += 1
                        dist += px
                    elif fdr[y, x] == 8:
                        x -= 1
                        y += 1
                        dist += px * 1.41  #Arrumar raiz aqui
                    elif fdr[y, x] == 16:
                        x -= 1
                        dist += px
                    elif fdr[y, x] == 32:
                        x -= 1
                        y -= 1
                        dist += px * 1.41  #Arrumar raiz aqui
                    elif fdr[y, x] == 64:
                        y -= 1
                        dist += px
                    elif fdr[y, x] == 128:
                        x += 1
                        y -= 1
                        dist += px * 1.41  #Arrumar raiz aqui

                    # elif fdr[y,x] == -100:
                    elif fdr[y, x] == 255:
                        # fdist[i,j] = -100
                        isnan = 1
                        break

                    loop += 1
                    if loop >= 500:
                        isnan = 1
                        break

                if isnan == 1:
                    indices[i, j] = -100
                    fdist[i, j] = -100
                else:
                    fdist[i, j] = dist
                    indices[i, j] = y * len(fdr[0]) + x

    return fdist, indices


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
    fdist : flow distance flot array, optional
        Flow distance. The default is np.array([]).

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

    # indices = np.zeros([len(fdr),len(fdr[0])],dtype=int32)
    indices = np.zeros(fdr.shape, int32)
    for i in range(0, len(fdr), 1):
        for j in range(0, len(fdr[0]), 1):
            if fdr[i, j] == 255:
                fdist[i, j] = -100
                indices[i, j] = -100
            elif new == 0 and fdist[i, j] != -50:
                continue
            else:
                y = i
                x = j
                dist = 0
                loop = 0
                isnan = 0
                while (river[y, x] != 1):
                    #alguma condição de fronteira aqui
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
                        dist += px * 1.41  #Arrumar raiz aqui
                    elif fdr[y, x] == 4:
                        y += 1
                        dist += px
                    elif fdr[y, x] == 8:
                        x -= 1
                        y += 1
                        dist += px * 1.41  #Arrumar raiz aqui
                    elif fdr[y, x] == 16:
                        x -= 1
                        dist += px
                    elif fdr[y, x] == 32:
                        x -= 1
                        y -= 1
                        dist += px * 1.41  #Arrumar raiz aqui
                    elif fdr[y, x] == 64:
                        y -= 1
                        dist += px
                    elif fdr[y, x] == 128:
                        x += 1
                        y -= 1
                        dist += px * 1.41  #Arrumar raiz aqui

                    # elif fdr[y,x] == -100:
                    elif fdr[y, x] == 255:
                        # fdist[i,j] = -100
                        isnan = 1
                        break

                    loop += 1
                    if loop >= 500:
                        isnan = 1
                        break

                if isnan == 1:
                    indices[i, j] = -100
                    fdist[i, j] = -100
                else:
                    fdist[i, j] = dist
                    indices[i, j] = y * len(fdr[0]) + x

    return fdist, indices


def flow_hand_index(dem, fdr, river, px, div_col=0, div_row=0):
    '''
    Overall method responsible for the matrix partitioning and
    defining dimensions.

    Parameters
    ----------
    dem : int or float array
        Digital elevation model raster.
    fdr : int8 array
        Flow direction.
    river : int8 array
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
    row = len(dem)
    col = len(dem[0])

    # Determinação das linhas e colunas de fronteira
    bRow, bCol = divisor(row, col, div_row, div_col)

    #Inicialização das matrizes
    fdist = np.zeros((row, col))
    indices = np.zeros((row, col), dtype='int')

    # now = time.time()

    #Se tiver divisão, cálculo dos índices e flow distance para a fronteira
    if div_row > 0 or div_col > 0:
        fdist[:, bCol] = -50
        fdist[bRow, :] = -50
        fdist, indices = fdist_indexes_sequential_jit(fdr, river, px, fdist)

    # print(time.time() - now)

    #Preenchimento para o dimensionamento das matrizes
    bRow = np.insert(bRow, div_row, row)
    bRow = np.insert(bRow, 0, -1)
    bCol = np.insert(bCol, div_col, col)
    bCol = np.insert(bCol, 0, -1)

    #Inicialização do apontador da parte
    part = 0

    for m in range(0, div_row + 1, 1):  # Linhas
        for n in range(0, div_col + 1, 1):
            #Array com direções de matrizes vizinhas
            out = np.zeros((4))
            #Arrays com as linhas e colunas de fronteira
            bound_c = np.zeros([1])
            bound_e = np.zeros([1])
            bound_d = np.zeros([1])
            bound_b = np.zeros([1])

            #Variáveis de início e fim da parte de matrzi
            mS = bRow[m]
            mE = bRow[m + 1]
            nS = bCol[n]
            nE = bCol[n + 1]

            #Preenchimento dos arrays de fronteira da vertical
            if div_row > 0:
                if part < (div_col + 1) * div_row:  #vizinho pra baixo
                    #como funciona:
                    #o part vai aumentar da esquerda para a direita,
                    #de cima para baixo. então desde que não esteja na última linha
                    #vai ter alguma área para baixo.
                    #exemplo: 2 div na coluna e uma na linha
                    #matriz de parte: [0,1,2] e [3,4,5]
                    #então: div_col+1 * div_row = 3*1 = 3
                    #desse modo, se part < 3, tem um pedaço da matriz para baixo
                    #Esse cálculo vai sempre resultar a posição da primeira parte da última linha
                    #Exemplo 4 div coluna e 2 row
                    # (divcol +1) * divrow = 5 * 2 = 10
                    # matriz: [0,1,2,3,4],[5,6,7,8,9],[10,11,12,13,14]
                    out[3] = 1  #Array de direções com matrizes
                    bound_b = fdist[mE, nS + 1:nE]
                    if n != div_col:
                        #Aqui, se n== div_col, então está na última coluna
                        #Se não está na última coluna, então vai ter uma parte para a direita
                        #nesse caso então a gente pega a célula em diagonal para baixo e direita
                        bound_b = np.insert(bound_b, nE - (nS + 1), fdist[mE,
                                                                          nE])
                    if n != 0:
                        #Mesma coisa para a esquerda
                        bound_b = np.insert(bound_b, 0, fdist[mE, nS])

                if part > div_col:  #vizinho em cima
                    out[0] = 1
                    bound_c = fdist[mS, nS + 1:nE]
                    if n != div_col:
                        bound_c = np.insert(bound_c, nE - (nS + 1), fdist[mS,
                                                                          nE])
                    if n != 0:
                        bound_c = np.insert(bound_c, 0, fdist[mS, nS])

            #Preenchimento dos arrays de fronteira horizontal
            if div_col > 0:
                if part % (div_col + 1) != 0:
                    #Aqui é para a esquerda. Se o módulo de part%(div_col+1) == 0
                    # então está na primeira coluna de partes e não tem matriz para a esquerda
                    out[1] = 1
                    bound_e = fdist[mS + 1:mE, nS]
                    if m != div_row:
                        #Célula diagonal
                        bound_e = np.insert(bound_e, mE - (mS + 1), fdist[mE,
                                                                          nS])
                    if m != 0:
                        #célula diagonal para cima
                        bound_e = np.insert(bound_e, 0, fdist[mS, nS])

                if part % (div_col + 1) != div_col:
                    out[2] = 1
                    bound_d = fdist[mS + 1:mE, nE]
                    if m != div_row:
                        bound_d = np.insert(bound_d, mE - (mS + 1), fdist[mE,
                                                                          nE])
                    if m != 0:
                        bound_d = np.insert(bound_d, 0, fdist[mS, nE])

            #Variavel para saber o tamanho do maior array para a inilização
            size = max(len(bound_c), len(bound_e), len(bound_d), len(bound_b))

            #Inicialização e preenchimento da matriz com os arrays de fronteira
            # bound_dist = np.zeros((4,size))
            bound = np.zeros((4, size))
            bound_index = np.zeros((4, size))
            bound[0, 0:len(bound_c)] = bound_c
            bound[1, 0:len(bound_e)] = bound_e
            bound[2, 0:len(bound_d)] = bound_d
            bound[3, 0:len(bound_b)] = bound_b

            mS += 1
            nS += 1
            part += 1

            # now = time.time()

            fdist[mS:mE, nS:nE], indices[mS:mE, nS:nE] = fdist_index_host(
                dem[mS:mE, nS:nE], fdr[mS:mE, nS:nE], river[mS:mE, nS:nE], px,
                bound, bound_index, out)
            # print(time.time() - now)
            # now = time.time()

            indices[mS:mE, nS:nE] = index_calculator(indices[mS:mE, nS:nE], mS,
                                                     nS, col)

            # print(time.time() - now)
    ##

    # now = time.time()
    hand = hand_calculator(dem, indices)
    # print(time.time() - now)

    return fdist, indices, hand
    # return fdist, indices


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


def hand_calculator_old(dem, indices):
    hand = np.zeros((len(dem), len(dem[0])))

    for i in range(0, len(dem), 1):
        for j in range(0, len(dem[0]), 1):
            if dem[i, j] == -100 or indices[i, j] == -100:
                hand[i, j] = -100
            else:
                ii = math.floor(indices[i, j] / len(indices[0]))
                jj = indices[i, j] % len(indices[0])
                jj = jj.astype('int')
                hand[i, j] = 0 if dem[i, j] - dem[ii, jj] < 0 else dem[
                    i, j] - dem[ii, jj]

    return hand


def index_calculator_old(ind, row_start, col_start, col_size):
    for i in range(0, len(ind), 1):
        for j in range(0, len(ind[0]), 1):
            if ind[i, j] != -100:
                print(i)
                print(j)
                return 10000
                # ind[i,j] = (math.floor(ind[i,j]/len(ind[0])) + row_start) * col_size + ind[i,j]%len(ind[0]) + col_start
    return ind
    #Lembrar que o seguinte, quando o indice sai do cuda e ele pegou de uma boundary ele vai pegar o valor de lá e o cálculo vai dar errado.


def index_calculator(ind, row_start, col_start, col_size):
    '''
    Method that transform the river index from the sub-matrix to the whole matrix

    Parameters
    ----------
    ind : int array
        Sub matrix river cell index.
    row_start : int
        Position of the first row of the sub-matrix.
    col_start : TYPE
        Position of the first column of the sub-matrix.
    col_size : TYPE
        Number of columns in the whole matrix.

    Returns
    -------
    ind : int array
        River cell index.

    '''
    row, col = ind.shape

    ind = np.where(ind == -100, -100,
                   (np.floor(ind / col) + row_start) * col_size + ind % col +
                   col_start)
    # ind = np.where(ind == -100,-100,(math.floor(ind/col) + row_start) * col_size + ind%col + col_start)

    return ind


def fdist_index_host(dem,
                     fdr,
                     river,
                     px,
                     bound_dist,
                     bound_index,
                     out,
                     blocks=0,
                     threads=0):
    '''
    Method responsible for the host/device data transfer

    Parameters
    ----------
    dem : int or float array
        Digital elevation model raster.
    fdr : int8 array
        Flow direction.
    river : int8 array
        Binary matrix that 1- river cell 0-not river cell.
    px : int or float
        Raster pixel dimension.
    bound_dist : float array
        Bounduary array that contains values of flow distance. 
        Necessary for flow directions that lead to outside of the sub-matrix domain.
    bound_index : int array
        Bounduary array that contains values of river indexes. 
        Necessary for flow directions that lead to outside of the sub-matrix domain.
    out : int array
        Array that represents wich directions have another sub-matrix.
        1-True 0-False. Up, left, right, down.
    blocks : int, optional
        Number of block of threads. The default is 0.
    threads : int, optional
        number of threads in each block. The default is 0.

    Returns
    -------
    fdist : float array
        Flow distance.
    indices : int array
        River cell index.

    '''
    row = len(dem)
    col = len(dem[0])

    if blocks == 0 and threads == 0:
        threads = 256
        blocks = math.ceil((row * col) / threads)
        #checar isso aqui

    dem = np.asarray(dem).reshape(-1)
    fdr = np.asarray(fdr).reshape(-1)
    river = np.asarray(river).reshape(-1)

    # hand = np.zeros((row*col), dtype='float32')
    fdist = np.zeros((row * col), dtype='float32')
    indices = np.zeros((row * col))

    fdr = cuda.to_device(fdr)
    fdist = cuda.to_device(fdist)
    indices = cuda.to_device(indices)
    out = cuda.to_device(out)
    bound_dist = cuda.to_device(bound_dist)
    bound_index = cuda.to_device(bound_index)

    fdist_index_device[blocks, threads](fdr, river, px, fdist, indices, col,
                                        row, bound_dist, bound_index, out)

    indices = indices.copy_to_host()
    fdist = fdist.copy_to_host()

    indices = indices.reshape(row, col)
    fdist = fdist.reshape(row, col)

    return fdist, indices


@cuda.jit
def fdist_index_device(fdr, river, px, fdist, indices, col, row, bound_dist,
                       bound_ind, out):
    '''
    GPU flow distance and river cell indexes method

    Parameters
    ----------
    dem : int or float array
        Digital elevation model raster.
    fdr : int8 array
        Flow direction.
    river : int8 array
        Binary matrix that 1- river cell 0-not river cell.
    px : int or float
        Raster pixel dimension.
    fdist : float array
        Flow distance.
    indices : int array
        River cell index.
    row : int
        Number of rows of the 2-D matrix.
    col : int
        Number of rows of the 2-D matrix.
    bound_dist : float array
        Bounduary array that contains values of flow distance. 
        Necessary for flow directions that lead to outside of the sub-matrix domain.
    bound_index : int array
        Bounduary array that contains values of river indexes. 
        Necessary for flow directions that lead to outside of the sub-matrix domain.
    out : int array
        Array that represents wich directions have another sub-matrix.
        1-True 0-False. Up, left, right, down.

    '''
    i = cuda.grid(1)
    if i >= 0 and i < col * row:
        if fdr[i] <= -100:
            fdist[i] = -100
            indices[i] = -100
        else:
            if river[i] == 1:
                fdist[i] = 0
                # indices[i] = 0
                # position = i
                indices[i] = i
            else:
                pos = i
                isnan = 0
                dist = 0
                loop = 0
                loop1 = 0
                loop2 = 0
                loop3 = 0
                while (river[pos] != 1):
                    #Bounduary conditions:
                    if pos < col and (fdr[pos] == 32 or fdr[pos] == 64
                                      or fdr[pos] == 128):
                        if out[0] == 0:
                            isnan = 1
                            break
                        else:
                            initial = pos
                            irow = int(initial)
                            if out[1] == 1:
                                irow += 1
                            if fdr[pos] == 32:

                                if pos == 0 and out[1] == 0:
                                    isnan = 1
                                    break
                                else:
                                    irow = irow - 1
                                    dist += (px * 1.4142) + bound_dist[0][irow]
                                    break
                            elif fdr[pos] == 128:
                                if pos == col - 1 and out[2] == 0:
                                    isnan = 1
                                    break
                                else:
                                    irow = irow + 1
                                    dist += px * 1.4142 + bound_dist[0][irow]
                                    break
                            else:
                                dist += px + bound_dist[0][irow]
                                break

                    elif pos % col == 0 and (fdr[pos] == 8 or fdr[pos] == 16
                                             or fdr[pos] == 32):
                        if out[1] == 0:
                            isnan = 1
                            break
                        else:
                            initial = pos / col
                            irow = int(initial)
                            if out[0] == 1:
                                irow += 1
                            if fdr[pos] == 32:
                                irow = irow - 1
                                dist += 1.4142 + bound_dist[1][irow]
                                break
                            elif fdr[pos] == 8:
                                if out[3] == 0 and pos == row * (col - 1):
                                    isnan = 1
                                    break
                                else:
                                    irow = irow + 1
                                    dist += px * 1.4142 + bound_dist[1][irow]
                                    break
                            else:
                                dist += px + bound_dist[1][irow]
                                break

                    elif pos % col == (col - 1) and (fdr[pos] == 128
                                                     or fdr[pos] == 1
                                                     or fdr[pos] == 2):
                        if out[2] == 0:
                            isnan = 1
                            break
                        else:
                            initial = (pos + 1) / col - 1
                            irow = int(initial)
                            if out[0] == 1:
                                irow += 1
                            if fdr[pos] == 128:
                                irow = irow - 1
                                dist += px * 1.4142 + bound_dist[2][irow]
                                break
                            elif fdr[pos] == 2:
                                if out[3] == 0 and pos == (row * col) - 1:
                                    isnan = 1
                                    break
                                else:
                                    irow = irow + 1  #Então eu pego um valor depois
                                    dist += px * 1.4142 + bound_dist[2][irow]
                                    break
                            else:
                                dist += px + bound_dist[2][irow]
                                break

                    elif pos >= (row - 1) * col and (fdr[pos] == 2 or fdr[pos]
                                                     == 4 or fdr[pos] == 8):
                        if out[3] == 0:
                            isnan = 1
                            break
                        else:
                            initial = pos - (row - 1) * col
                            irow = int(initial)
                            if out[1] == 1:
                                irow += 1
                            if fdr[pos] == 8:
                                irow = irow - 1
                                dist += px * 1.4142 + bound_dist[3][irow]
                                break
                            elif fdr[pos] == 2:
                                irow += 1
                                dist += px * 1.4142 + bound_dist[3][irow]
                                break
                            else:
                                dist += px + bound_dist[3][irow]
                                break
                    ######

                    #loop update
                    loop3 = loop2
                    loop2 = loop1
                    loop1 = pos
                    ######

                    #### Next cell direciton
                    if fdr[pos] == 1:
                        pos += 1
                        dist += px
                    elif fdr[pos] == 2:
                        pos += 1 + col
                        dist += (px * 1.4142)
                    elif fdr[pos] == 4:
                        pos += col
                        dist += px
                    elif fdr[pos] == 8:
                        pos += col - 1
                        dist += (px * 1.4142)
                    elif fdr[pos] == 16:
                        pos += -1
                        dist += px
                    elif fdr[pos] == 32:
                        pos += -1 - col
                        dist += (px * 1.4142)
                    elif fdr[pos] == 64:
                        pos += -col
                        dist += px
                    elif fdr[pos] == 128:
                        pos += -col + 1
                        dist += (px * 1.4142)

                    if fdr[pos] == -100:
                        isnan = 1
                        break
                    ####

                    #Loop conditions
                    if pos == loop1 or pos == loop2 or pos == loop3:
                        isnan = 1
                        break

                    loop += 1
                    if loop > 20000:
                        isnan = 1
                        break
                    #####

                if isnan == 1:
                    fdist[i] = -100
                    indices[i] = -100
                else:
                    fdist[i] = dist
                    indices[i] = pos
