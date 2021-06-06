# Author: HTC

# This function generates a_mat such that A_mat @ B_mat = 1
# Inputs: b_mat: 2D array, number of stragglers (s): int, number of clients (n): int
# Outputs: a_mat: 2D array

import itertools
import numpy as np
import math as m


def matrix_a(b_mat, n, s, debug=0):
    # Create all the possible subsets
    n_list = list(range(0, n))
    length_subset = n - s
    sub_list = list(itertools.combinations(n_list, length_subset))
    # Solve for Matrix A s.t. AB = 1 (f x k)
    f = m.comb(n, s)
    a_mat = np.zeros((f, n))
    for i in n_list:
        if debug:
            print("i: ", i)
        a = np.zeros((1, n))[0]
        ones = np.ones(n)
        if debug:
            print(np.asarray(sub_list[i]))
        rows_b = b_mat[np.asarray(sub_list[i])]
        if debug:
            print("rows_b: \n", rows_b)
        rows_b_t = np.transpose(rows_b)
        if debug:
            print(np.linalg.matrix_rank(rows_b))
            print(np.linalg.matrix_rank(rows_b_t))
            print(np.linalg.matrix_rank(ones))
            print("rows_b trans: \n", rows_b_t)
            print(np.shape(rows_b_t), np.shape(ones))
        x = np.linalg.lstsq(rows_b_t, ones, rcond=None)
        if debug:
            print("x: ", x)
            print("rows_b_t @ x: \n", rows_b_t @ x[0])
        a[np.asarray(sub_list[i])] = x[0]
        if debug:
            print("a: \n", a)
        a_mat[n - 1 - i] = a

    # print("a_mat: \n", a_mat)
    # print("b_mat: \n", b_mat)
    # print("a_mat @ b_mat: \n", a_mat @ b_mat)
    return a_mat

# sample code
# b_mat = np.asarray([[.5, 1, 0], [0, 1, -1], [0.5, 0, 1]])
# n = 3
# s = 1
# a_mat(b_mat, n, s)
