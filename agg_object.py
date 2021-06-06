# Author: HTC 10.15.20
# This is the main script for an Aggregator
# The Agg object performs the following:
# 1. Constructs exhaustive failure scenarios
# 2. Creates data partitions
# 3. Sends and receives data from nodes
# 4. Updates global model (?)

from matrix_b_fr import matrix_b_fr
from matrix_b_cyc import matrix_b_cr
from matrix_a import matrix_a
from aggr_client_v1_3 import client
from aggr_server import aggr_server as server
from random_linear_data import *
import numpy as np
import threading
import queue
from matrix_operations import *
import time


# rp_addresses defines the number of clients participating in the experiment including stragglers
laptop_address = '192.168.0.6'
rp_addresses = ['192.168.0.18', '192.168.0.17', '192.168.0.19', '192.168.0.30']
stragglers = ['192.168.0.30']


class Agg:
    # Constructor
    def __init__(self, fr=1):

        # This queue holds information that we need to send
        self._sendingQueue = queue.Queue(maxsize=40)

        # Threads for sending and receiving information from the other nodes
        self._sendingThread = threading.Thread(target=self.sendingLoop)
        self._sendingThread.start()
        self._receivingThread = threading.Thread(target=self.receivingLoop)
        self._receivingThread.start()

        # Use the matrix functions to define the gradient coding scheme
        # B specifies which data partitions to send to each node
        # A specifies the linear combination to get back all gradients
        self._n = len(rp_addresses)
        self._s = len(stragglers)
        if fr == 1:
            b_mat = matrix_b_fr(self._n, self._s)
            print("b_mat: \n", b_mat)
        else:
            b_mat = matrix_b_cr(self._n, self._s, debug=0)

        # Generate some random data
        x, y = genData(1000, 25, 5)
        print(x)
        print(y)

        # Partition data into the appropriate number of partitions (number of workers)
        # Each partition will have 1000 rows and 1000/n cols (for 1000 x 1000 data set)

        partitioned_data = np.array_split(y, self._n, axis=1)
        print("shape of partitions: ", np.shape(partitioned_data[0]))

        # Build a matrix containing the proper partitions (from mat_b)
        # Assumes only two partitions are assigned per worker (11/04/20)
        for i in range(self._n-1):
            # create a matrix to send to a worker containing the correct partitions
            partition_loc = np.nonzero(b_mat[i])[0]      # locations of non-zero entries for one row of b_mat
            print("partitions for worker %d : " % i, partition_loc)
            self._matrix = matrix_merge(partitioned_data[partition_loc[0]], partitioned_data[partition_loc[1]], 1)

            # add the x col vector to the front of the matrix
            self._matrix = matrix_merge(x, self._matrix, 1)

            # add an info column at the front
            self._matrix = matrix_merge(np.ones((int(len(y)), 1)), self._matrix, 1)
            self._matrix[0, 0] = len(partition_loc)     # How many partitions are included in this matrix
            self._matrix[1, 0] = i                      # index of RP ip_address from RP_address list
            print("Added info col. Final shape of matrix: \n", np.shape(self._matrix))
            print("matrix: \n", self._matrix[0:5, 0:4])
            self._sendingQueue.put(self._matrix)

        # Code to decipher gradient coding
        a_mat = matrix_a(b_mat, self._n, self._s, debug=0)
        a_row = a_mat[rp_addresses.index(stragglers[0])]  # assumes there is only one straggler
        print("a_row: ", a_row)

    # A loop to send messages to aggregator

    def sendingLoop(self):
        while True:
            if not self._sendingQueue.empty():
                print("Sending...")
                tup = self._sendingQueue.get()
                client(rp_addresses[int(tup[1, 0])], tup)
            time.sleep(3)
    print("Finished Sending")

    # A loop to receive messages from nodes

    def receivingLoop(self):
        while True:
            item = server(laptop_address, self._n - self._s)
            print("received data from nodes")
            print("item: \n", item)
            print("Shape of received item:")
            print(np.array(item).shape)
            time.sleep(3)


if __name__ == "__main__":
    agg = Agg()
