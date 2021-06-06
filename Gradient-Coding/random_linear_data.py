# Generates some random linear data
# outputs an x vector (1:num_points) and a y vector (random data)
import numpy as np
import random


def genData(numPoints, bias, variance):
    x = np.zeros(shape=(numPoints, 1))
    y = np.zeros(shape=(numPoints, numPoints))
    for i in range(0, numPoints):
        # x values
        x[i][0] = i

        # random y values
        for j in range(0, numPoints):
            y[i, j] = (i + bias) + random.uniform(0, 1) * variance
    return x, y
