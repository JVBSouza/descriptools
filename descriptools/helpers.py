import numpy as np
import math


def divisor(row_length, column_length, row_division, column_division):
    boundary_column = np.array([], dtype=int)
    boundary_row = np.array([], dtype=int)

    for i in range(0, row_division, 1):
        boundary_row = np.append(
            boundary_row,
            [math.floor((i + 1) * row_length / (row_division + 1))])
    for i in range(0, column_division, 1):
        boundary_column = np.append(
            boundary_column,
            [math.floor((i + 1) * column_length / (column_division + 1))])

    return boundary_row, boundary_column