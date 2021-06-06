# This script generates a random matrix based on eigenvalue decomposition
# based off Mingyue Ji's Uncoded_CEC_Power_Iteration.py file

import numpy as np
from matrix_operations import partition

def build_random_matrix(length, eig_values=np.array([1, 0.94, 0.7, 0.6, 0.5])):
    num_eigs = len(eig_values)

    # if num_eigs != length:
        # eig_values = np.append(eig_values, np.zeros(length - num_eigs))
        # print(eig_values)
        # num_eigs = len(eig_values)

    # Gram-Schmidt process to create orthogonal eigen vectors
    B = np.random.rand(num_eigs, length)  # used to define eigen vectors
    # print(B)
    for i in range(num_eigs):
        for j in range(i):
            B[i] -= B[j] * np.dot(B[j], B[i]) / np.linalg.norm(B[j])
        B[i] = B[i] / np.linalg.norm(B[i])

    # Multiply matrices to build final matrix
    # print('B: ', np.shape(B), 'Diag: ', np.shape(np.diag(eig_values)))
    mat = np.transpose(B) @ np.diag(eig_values) @ B
    # mat = np.matmul(np.diag(eig_values), B)
    return mat, np.transpose(B)

#############
# Test Code #
#############


eig_vals = [1, 0.95, 0.8, 0.7, 0.6]
rand, eig_vect = build_random_matrix(1200, eig_values=eig_vals)

print("Rand: \n", rand, "\nShape of Rand: ", np.shape(rand))

# Find eigen vectors
# eig_vals, eig_vect = np.linalg.eig(rand)

index = np.where(eig_vals == np.max(eig_vals))[0][0]
print('max calculated eigen values: ', np.max(eig_vals))
print('loc of max eigen value: ', index)
print(eig_vals)
print(eig_vals[index])
print(np.shape(eig_vect))



b = np.ones((np.shape(rand)[0]))
print('Starting b: ', b)
num_iter = range(200)

for i in num_iter:
    b_temp = np.dot(rand, b)
    # print('b_temp: ', b_temp)

    b_norm = np.linalg.norm(b_temp)
    # print('b_norm: ', b_norm)

    b = b_temp / b_norm
    print('b: ', b)

    error = np.linalg.norm(b - eig_vect[:, index]) / np.linalg.norm(eig_vect[:, index])
    # print('Error: ', error)
    # print()

# print('Eigen values: ', eig_vals)
# print('Eigen vectors: ', eig_vect)
# print('Max Eigen Val: ', eig_vals[-1])
print('b: ', b, 'shape of b: ', np.shape(b))
print('Error for opp b: ', np.linalg.norm(b*-1 - eig_vect[:, index]) / np.linalg.norm(eig_vect[:, index]))
print('Error for b: ', np.linalg.norm(b - eig_vect[:, index]) / np.linalg.norm(eig_vect[:, index]))
# print('Mean Square Error', mean_squared_error(y_true=eig_vect[:, index], y_pred = np.reshape(b,(2,1))))
print('Opposite of b: ', b*-1)
print('Max Eigen vect: ', eig_vect[:, index])

# do a comparison between b and the max eigen vector
no_wrong = 0
tol = .1
for count, j in enumerate(eig_vect[:, index]):
    if abs(j - b[count]) >= tol:
        # print("Found a wrong value")
        no_wrong += 1
print('no_wrong: ', no_wrong)

# do a comparison between b*-1 and the max eigen vector
no_wrong_opp = 0
for count, j in enumerate(eig_vect[:, index]):
    if abs(j - b[count]*-1) >= tol:
        # print("Found a wrong value")
        no_wrong_opp += 1
print('no_wrong_opp: ', no_wrong_opp)


'''
# Test uncoded version
num_iter = 150
n = 3
length = 1200
eig_vals = [1, 0.95, 0.8, 0.7, 0.6]
rand, eig_vect = build_random_matrix(length, eig_values=eig_vals)
index = np.where(eig_vals == np.max(eig_vals))
max_vect = eig_vect[:, index]
print(np.shape(max_vect))

vect_init = np.ones((length, 1))
vect = partition(vect_init, n, by_rows=1, vect=1)
print(np.shape(vect))
parts = partition(rand, n, by_rows=0)
print(np.shape(parts))

for i in range(num_iter):
    if i == 0:
        vect_1 = parts[0][:][:] @ vect[0][:][:]
        vect_2 = parts[1][:][:] @ vect[1][:][:]
        if n == 3:
            vect_3 = parts[2][:][:] @ vect[2][:][:]
            print(vect_1, '\n', vect_2, '\n', vect_3)
            b = vect_1 + vect_2 + vect_3
            print(b)
        else:
            print(vect_1, '\n', vect_2, '\n')
            b = vect_1 + vect_2
            print(b)

        b_norm = np.linalg.norm(b)
        print(b_norm)
        b_new = b / b_norm
        error = np.linalg.norm(b_new - max_vect[:, :, 0]) / np.linalg.norm(max_vect[:, :, 0])
    else:
        vect = partition(b_new, n, by_rows=1, vect=1)
        if n == 3:
            vect_1 = parts[0][:][:] @ vect[0][:][:]
            vect_2 = parts[1][:][:] @ vect[1][:][:]
            vect_3 = parts[2][:][:] @ vect[2][:][:]
            print(vect_1, '\n', vect_2, '\n', vect_3)

            b = vect_1 + vect_2 + vect_3
            print(b)
        else:
            vect_1 = parts[0][:][:] @ vect[0][:][:]
            vect_2 = parts[1][:][:] @ vect[1][:][:]
            print(vect_1, '\n', vect_2)
        b_norm = np.linalg.norm(b)
        print(b_norm)
        b_new = b / b_norm
        error = np.linalg.norm(b_new - max_vect[:, :, 0]) / np.linalg.norm(max_vect[:, :, 0])
print('Final error: ', error)
print(b_new)
print(max_vect)
'''