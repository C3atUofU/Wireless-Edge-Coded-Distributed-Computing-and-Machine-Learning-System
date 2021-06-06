# Adapted from MCG's node.py
# Additions by HTC
# Creates an object that can perform all tasks for a worker node

import queue
import threading
import time
import numpy as np
from matrix_operations import *
from node_client import client
from node_server import server

# Initialize global variables
laptop_address = '192.168.0.6'


class Node:

    # CONSTRUCTOR
    def __init__(self):
        # Define the device's IP_address.
        self._ipAddr = '192.168.0.18'   # update on each node locally

        # when a matrix is ready for grad calc toggle to True
        self._matrixReady = False

        # initialize the private variables that will be used in sending, receiving, and multiplying
        self._g = []
        self._no_partitions = 0
        self._x = []
        self._matrix = []
        self._my_num = 0

        # This queue holds information that we need to send
        self._sendingQueue = queue.Queue(maxsize=40)

        # Threads for sending and receiving information from the other nodes
        self._sendingThread = threading.Thread(target=self.sendingLoop)
        self._sendingThread.start()
        self._receivingThread = threading.Thread(target=self.receivingLoop)
        self._receivingThread.start()

        # Thread for doing the matrix gradient work
        self._gradThread = threading.Thread(target=self.multLoop)
        self._gradThread.start()

    # A loop to send messages to aggregator

    def sendingLoop(self):
        while True:
            if not self._sendingQueue.empty():
                print("Sending...")
                tup = self._sendingQueue.get()
                client(tup[0], tup[1], laptop_address)
            time.sleep(3)

    # A loop to receive messages from aggregator

    def receivingLoop(self):
        while True:
            item = server(self._ipAddr)
            print("Updating based on received information...")
            print("item: \n", item)
            print("Shape of received item:")
            print(np.array(item).shape)

            # assume that item contains multiple partitions of same size
            # First column will contain general information about item received (e.g. how many partitions)
            self._no_partitions = int(item[0, 0])
            self._my_num = int(item[1, 0])
            self._x = item[:, 1]
            self._matrix = item[:, 2:]
            self._matrixReady = True
            time.sleep(3)

    # Calculate matrix multiplication for each partition

    def multLoop(self):
        while True:
            # do nothing until a matrix is received
            if self._matrixReady:
                # calculate the gradient for each partition
                mat_1, mat_2 = matrix_split(self._matrix)
                g_1 = np.matmul(np.transpose(self._x), mat_1)
                print("g_1: \n", g_1)
                print(np.shape(g_1))
                g_2 = np.matmul(np.transpose(self._x), mat_2)
                print("g_2: \n", g_2)
                print(np.shape(g_2))

                # merge the gradients together then send back
                self._g = matrix_merge(g_1, g_2, 0)
                self._g = matrix_merge(np.ones((int(len(self._g))), 1, self._g, 1))  # add rp index to send data
                print("self._g: \n", self._g)
                print(np.shape(self._g))
                self._sendingQueue.put(self._g)
                self._matrixReady = False
            else:
                time.sleep(1)


if __name__ == "__main__":
    node = Node()
