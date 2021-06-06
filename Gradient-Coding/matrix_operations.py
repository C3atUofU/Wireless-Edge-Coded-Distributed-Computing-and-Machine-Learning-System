# Houses some of the frequent matrix operations the node and agg will need to perform
# HTC 11/09/20

import numpy as np


# matrix merge operation
# Inputs: mat_1, mat_2, and axis (axis=1 merges about columns, axis=0 merges about rows)
# Note that mat_1 and mat_2 must have the same shape except in the dimension corresponding to the axis
# Outputs: matrix, the merged form of mat_1 and mat_2
def matrix_merge(mat_1, mat_2, axis):
    matrix_out = np.concatenate((mat_1, mat_2), axis=axis)
    print("matrix merged. Final dimensions: \n", np.shape(matrix_out))
    return matrix_out


# matrix split operation using array slicing. Slices by rows. Only splits into 2 matrices
# Inputs: a matrix (e.g. 1000 x 1000)
# Outputs: mat_1 and mat_2 each of identical dimensions (e.g. 500 x 1000)
def matrix_split(matrix, rows=1):
    if rows == 1:
        mat_1 = matrix[0:int(np.shape(matrix)[0] / 2), :]
        mat_2 = matrix[int(np.shape(matrix)[0] / 2):int(np.shape(matrix)[0]), :]
        print("Matrix split in 2 matrices of shape: \n", np.shape(mat_1), np.shape(mat_2))
    else:
        mat_1 = matrix[:, 0:int(np.shape(matrix)[1] / 2)]
        mat_2 = matrix[:, int(np.shape(matrix)[1] / 2):int(np.shape(matrix)[1])]
        print("Matrix split in 2 matrices of shape: \n", np.shape(mat_1), np.shape(mat_2))
    return mat_1, mat_2


# Partitions a matrix evenly by either rows or cols
# INPUTS: matrix to partition, the number of partitions to create, a switch for row-wise or col-wise
# OUTPUTS: a 3d matrix with partitions represented by each index
def partition(matrix, no_parts, by_rows=0, vect=0):
    if vect:
        if np.shape(matrix)[1] > 1:
            # print("Row Vector")
            length = np.shape(matrix)[1]
            if by_rows:
                print("Error: Can't partition row vector row-wise")
                exit()
        elif np.shape(matrix)[0] > 1:
            # print("Column Vector")
            length = np.shape(matrix)[0]
            if not by_rows:
                print("Error: Can't partition column vector column-wise")
                exit()
        else:
            print("Error: Can't partition 1 x 1 vector")
            exit()
    else:
        length = len(matrix)
    # print(length)
    mat_out = []
    counter = 0
    if not by_rows:
        # partition column-wise
        for ii in range(no_parts):
            # print("index ", ii)
            temp = matrix[:, int(counter * length / no_parts): int((counter + 1) * length / no_parts)]
            # print("temp \n", temp)
            # print("shape of temp: ", np.shape(temp))
            mat_out.append(temp)
            counter += 1
            # print(mat_out)
    else:
        # partition row-wise
        for ii in range(no_parts):
            # print("index ", ii)
            temp = matrix[int(counter * length / no_parts): int((counter + 1) * length / no_parts), :]
            # print("temp \n", temp)
            # print("shape of temp: ", np.shape(temp))
            mat_out.append(temp)
            counter += 1
            # print(mat_out)
    return mat_out


"""
# Code test
vector = np.random.randint(0, 100, (1200, 1))
vect_out = partition(vector, 3, by_rows=1, vect=1)
print("Output: ", vect_out, "Shape of output: ", np.shape(vect_out))
print("First part: ", vect_out[0], "\nShape of part: ", np.shape(vect_out[0]))
"""
