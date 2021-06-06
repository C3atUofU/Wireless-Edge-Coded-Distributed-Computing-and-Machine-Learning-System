# Author: HTC

# Creates the B matrix for n workers and s stragglers, such that n is a # multiple of (s+1)
# Inputs: number of clients (n), number of stragglers (s)
# Outputs: b_fract representing which data partitions are available to each worker

import numpy as np
from matrix_a import *
run_example = False


def matrix_b_fr(n=4, s=1):
    # Number of workers must be a multiple of (s+1)
    if n % (s + 1) != 0:
        print("number of workers (n) must be a multiple of number of stragglers (s)!")
        exit()

    # Define Groups
    no_of_groups = s + 1
    workers_per_group = int((n / (s + 1)))

    # Build a matrix for each group of workers. dimension = workers_per_group x n
    # first worker in the group gets the first (s+1) partitions, and so on
    dimension_of_b = (workers_per_group, n)
    b_group = np.zeros(dimension_of_b)
    for i in range(workers_per_group):
        b_group[i, i * (s + 1):(1 + i) * (s + 1)] = 1

    # Duplicate the b_group matrix s+1 times to create the b_fract matrix
    for i in range(no_of_groups):
        if i == 0:
            b_fract = b_group
        else:
            b_fract = np.vstack((b_fract, b_group))
    return b_fract


if run_example:
    n = 4
    s = 1
    b_mat = matrix_b_fr(n, s)
    a_mat = matrix_a(b_mat, n, s)
    print("a_mat: \n", a_mat)
    print("a_mat @ b_mat: \n", a_mat @ b_mat)
    print(b_mat)
