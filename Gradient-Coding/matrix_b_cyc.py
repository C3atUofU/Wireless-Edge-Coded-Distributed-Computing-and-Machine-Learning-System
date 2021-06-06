# Author: HTC 10/27/20

# This program generates a b_mat using a Cyclic Repetition Scheme
# Each row of b_mat will contain s+1 non-zero elements

import numpy as np
from matrix_a import matrix_a
run_example = False


def matrix_b_cr(n=3, s=1, debug=0):
    H = np.random.standard_normal(size=(s, n))
    if debug:
        print("H: \n", H)
        print("Last col of H: \n", H[:, n-1])
        print(H[:, 0:n-1])
    H[:, n-1] = -np.sum(H[:, 0:n-1])
    if debug:
        print(H)
    b_mat = np.zeros((n, n))
    for i in range(0, n):
        if debug:
            print("i: ", i)
        if debug:
            print(np.array((i, s+i)))
        j = np.mod([i, s+i], n)
        if debug:
            print("j: ", j)
            print("j_sub: ", j[1:s+1])
            print("b_mat: ", b_mat[i, j])
            print("H_1: ", -H[:, j[1:s+1]][0])
            print("H_2: ", H[:, j[0]])
            print(np.linalg.lstsq(-H[:, j[1:s + 1]], H[:, j[0]], rcond=None))
        b_mat[i, j] = [1, np.linalg.lstsq(-H[:, j[1:s+1]], H[:, j[0]], rcond=None)[0]]
    return b_mat


if run_example:
    n = 3
    s = 1
    b_mat_cr = matrix_b_cr(n, s, debug=1)
    a_mat = matrix_a(b_mat_cr, n, s, debug=1)
    print("b_mat: \n", b_mat_cr)
    print("a_mat: \n", a_mat)
    print("a_mat @ b_mat: \n", a_mat @ b_mat_cr)

