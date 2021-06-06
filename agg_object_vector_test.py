# Author: HTC 10.15.20
# This is the main script for an Aggregator using a simple vector
# The Agg object performs the following:
# 1. Constructs exhaustive failure scenarios
# 2. Generates a random data vector
# 3. Sends and receives data from nodes


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
rp_addresses = ['192.168.0.17', '192.168.0.18', '192.168.0.19']
stragglers = ['192.168.0.19']


class Agg:
    # Constructor
    def __init__(self):
        # This queue holds information that we need to send
        self._sendingQueue = queue.Queue(maxsize=40)

        # Initialize variables
        self._item = []         # variable for received data
        self._length = 100000     # variable for length of dataset
        # Threads for sending and receiving information from the other nodes
        self._sendingThread = threading.Thread(target=self.sendingLoop)
        self._sendingThread.start()
        self._receivingThread = threading.Thread(target=self.receivingLoop)
        self._receivingThread.start()

        # Generate a random data matrix, each worker will receive one column of the matrix
        self._n = len(rp_addresses)
        self._s = len(stragglers)
        print("n: ", self._n)
        print("s: ", self._s)
        self._rand_data = np.random.standard_normal((self._length+1, 3))
        print("rand_data: \n", self._rand_data, "\n shape of rand_data: ", np.shape(self._rand_data))

        # Manually assign data partitions as specified in Fig 1 of UT Austin paper
        self._worker_1 = self._rand_data[:, [0, 1]]
        self._worker_2 = self._rand_data[:, [1, 2]]
        self._worker_3 = self._rand_data[:, [2, 0]]
        self._worker_1[0, 0] = 0
        self._worker_1[0, 1] = 0
        self._worker_2[0, 0] = 1
        self._worker_2[0, 1] = 1
        self._worker_3[0, 0] = 2
        self._worker_3[0, 1] = 2
        print("Worker 1: \n", self._worker_1[0:8, :], "\n shape of worker 1: ", np.shape(self._worker_1))
        print("Worker 2: \n", self._worker_2[0:8, :], "\n shape of worker 2: ", np.shape(self._worker_2))
        print("Worker 3: \n", self._worker_3[0:8, :], "\n shape of worker 3: ", np.shape(self._worker_3))
        self._g1 = np.ones((1, self._length)) @ np.reshape(self._rand_data[1:, 0], (self._length, 1))
        self._g2 = np.ones((1, self._length)) @ np.reshape(self._rand_data[1:, 1], (self._length, 1))
        self._g3 = np.ones((1, self._length)) @ np.reshape(self._rand_data[1:, 2], (self._length, 1))
        self._local_grad = self._g1 + self._g2 + self._g3

        # Place worker vectors in the queue
        self._sendingQueue.put(self._worker_1)
        self._sendingQueue.put(self._worker_2)

    # A loop to send messages to aggregator

    def sendingLoop(self):
        while True:
            if not self._sendingQueue.empty():
                tup = self._sendingQueue.get()
                print("Sending to worker", int(tup[0, 0]))
                client(rp_addresses[int(tup[0, 0])], tup)
                print("Finished Sending")
            time.sleep(3)


    # A loop to receive messages from nodes

    def receivingLoop(self):
        while True:
            self._item = server(laptop_address, 2)  # Update from 2 to (self._n - self._s)
            print("received data from nodes")
            print("item: \n", self._item)
            print("item.fn:", self._item.fn, "\nitem.w", self._item.w)
            # fn tells us which row belongs to which worker (i.e. if first entry in item.fn is zero, that means the
            # first row of item.w is from worker 0.  Will use to acquire full gradient

            # This algorithm comes from UT Austin paper fig. 1
            grad = 2*(self._item.w[0, 0]/2 + self._item.w[0, 1]) - (self._item.w[1, 0] - self._item.w[1, 1])
            print("g1 local: ", self._g1[0, 0])
            print("g2 local: ", self._g2[0, 0])
            print("g3 local: ", self._g3[0, 0])
            print("g1 + g2 + g3 = ", grad)
            print("locally calculated grad: ", self._local_grad[0, 0])
            if abs(grad - self._local_grad[0, 0]) < 0.000001:
                print("GREAT SCOTT!!! (it works)")
            time.sleep(3)


if __name__ == "__main__":
    agg = Agg()
