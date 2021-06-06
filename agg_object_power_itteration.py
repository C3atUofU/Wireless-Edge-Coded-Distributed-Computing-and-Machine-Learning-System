# Author: HTC 10.15.20
# Uses agg and node to perform a power iteration
# 1. "Random" data matrix sent to workers
# 2. workers compute gradient vectors send to agg
# 3. Agg combines gradient vectors, normalizes then sends back to nodes
# 4. Process repeats until iteration limit hit
# 5. Error and time to complete calculated at each step

from matrix_b_fr import matrix_b_fr
from matrix_b_cyc import matrix_b_cr
from matrix_a import matrix_a
from save_to_log import *
from aggr_client_v1_3 import client
from aggr_server import aggr_server as server
from eigen_decomp import build_random_matrix
from agg_plots import *
import numpy as np
import threading
import queue
from matrix_operations import matrix_merge, matrix_split, partition
import time


# Program timer
start = time.time()


class Agg:
    # Constructor
    def __init__(self):
        # beginning time stamp
        time_init_start = time.time()

        # rp_addresses defines the number of clients participating in the experiment including stragglers
        self._laptop_address = '192.168.0.28'
        self._rp_addresses = ['192.168.0.18', '192.168.0.19', '192.168.0.17']
        self._stragglers = ['192.168.0.17']

        # This queue holds information that we need to send
        self._sendingQueue = queue.Queue(maxsize=40)

        # Initialize variables
        self._item = []         # variable for received data
        self._length = 1200      # variable for length of dataset
        self._iterations = 10  # variable to specify how many iterations to carry out
        self._count = 0         # variable for tracking how many iterations have been completed
        self._time_init = 0
        self._time_send_list = []
        self._time_receive_list = []
        self._time_action_list = []
        self._error_list = []
        self._ready_to_process = False
        self._max_eigen = []
        self._index = 0
        self._tol = .01     # threshold for determining an incorrect value in final vect comparison

        # Threads for sending, receiving and processing information from the other nodes
        self._sendingThread = threading.Thread(target=self.sending_loop)
        self._sendingThread.start()
        self._receivingThread = threading.Thread(target=self.receiving_loop)
        self._receivingThread.start()
        self._actionThread = threading.Thread(target=self.action_loop)
        self._actionThread.start()

        # Generate a random data matrix
        self._n = len(self._rp_addresses)
        self._s = len(self._stragglers)
        print("n: ", self._n)
        print("s: ", self._s)
        eig_vals = np.array([1, 0.8, 0.7, 0.6])
        self._rand_data, eig_vect = build_random_matrix(self._length, eig_vals)
        print("rand_data: \n", self._rand_data[0:6, 0:6], "\n shape of rand_data: ", np.shape(self._rand_data))
        self._index = np.where(eig_vals == np.max(eig_vals))
        self._max_eigen = eig_vect[:, self._index]

        # Partition random dataset column-wise
        parts = partition(self._rand_data, self._n, by_rows=0)
        print("partition 1: ", parts[0][0:5, 0:5], "\nShape of partition 1: ", np.shape(parts[0]))
        print("partition 2: ", parts[1][0:5, 0:5], "\nShape of partition 2: ", np.shape(parts[1]))
        print("partition 3: ", parts[2][0:5, 0:5], "\nShape of partition 3: ", np.shape(parts[2]))

        # Manually assign data partitions as specified in Fig 1 of UT Austin paper
        self._worker_1 = matrix_merge(parts[0], parts[1], 1)
        self._worker_2 = matrix_merge(parts[1], parts[2], 1)
        self._worker_3 = matrix_merge(parts[2], parts[0], 1)
        print("Worker 1: \n", self._worker_1[0:10, 0:10], "\n shape of worker 1:", np.shape(self._worker_1))
        print("Worker 2: \n", self._worker_2[0:8, 0:8], "\n shape of worker 2: ", np.shape(self._worker_2))
        print("Worker 3: \n", self._worker_3[0:8, 0:8], "\n shape of worker 3: ", np.shape(self._worker_3))

        # Place worker vectors in the queue (only to two, b/c third is assumed to be straggler)
        self._sendingQueue.put(self._worker_1)
        self._sendingQueue.put(self._worker_2)

        # Time stamp
        time_init_end = time.time()
        self._time_init = time_init_end - time_init_start
        print("Constructor time: ", self._time_init)

    # A loop to send messages to aggregator
    def sending_loop(self):
        index = 0
        self._done_sending = False
        while True:
            if not self._sendingQueue.empty():
                send_time_start = time.time()
                tup = self._sendingQueue.get()
                print("Sending to worker", index)
                client(self._rp_addresses[index], tup)
                print("Finished Sending")
                index += 1
                print("index: ", index)
                if index == (self._n - self._s):
                    print("Index reset")
                    index = 0
                    self._done_sending = True
                # Time stamp
                send_time_end = time.time()
                self._time_send_list.append(send_time_end - send_time_start)
                print("Sending iteration time: ", send_time_end - send_time_start)
            time.sleep(3)

    # A loop to receive messages from nodes
    def receiving_loop(self):
        while True:
            receive_time_start = time.time()
            self._item = server(self._laptop_address, 2, self._length)  # Update from 2 to (self._n - self._s)
            print("received data from nodes")
            # print("item.fn:", self._item.fn, "\nitem.w", self._item.w)
            self._ready_to_process = True

            # Time stamp
            receive_time_end = time.time()
            self._time_receive_list.append(receive_time_end - receive_time_start)
            print("Receiving iteration time: ", receive_time_end - receive_time_start)
            time.sleep(0.5)

    # Loop to process data
    def action_loop(self):
        while True:
            if self._ready_to_process:
                start_time = time.time()
                # This if else loop handles the order the RPs respond
                # fn tells us which col belongs to which worker (i.e. if first entry in item.fn is zero, that means the
                # first col of item.w is from worker 0.  Will use to acquire full gradient
                if self._item.fn[0] == self._rp_addresses[0]:
                    w1 = 0
                    w2 = 2
                else:
                    w1 = 2
                    w2 = 0

                # recover the full gradient sum using eq in fig. 1 of UT Austin paper
                grad_sum = 2 * (self._item.w[:, w1] / 2 + self._item.w[:, w1 + 1]) - (
                        self._item.w[:, w2] - self._item.w[:, w2 + 1])
                print("\ng_1 + g_2 + g_3 = \n", grad_sum, "\nShape of sum: ", np.shape(grad_sum))
                b = np.reshape(grad_sum, (np.shape(grad_sum)[0], 1))

                # Normalize
                b_norm = np.linalg.norm(b)
                b_new = b / b_norm
                print("b_k+1 = \n", b_new[0:10, :], "\nShape of new vector: ", np.shape(b_new))

                # Calculate the error of this step
                error = np.linalg.norm(b_new - self._max_eigen[:, :, 0]) / np.linalg.norm(self._max_eigen[:, :, 0])
                print("\nError: ", error)
                self._error_list.append(error)

                # partition and merge vector to be sent to nodes
                vector_parts = partition(b_new, self._n, by_rows=1, vect=1)
                print("vect part 1: ", vector_parts[0][0:10, :], "\n shape of part: ", np.shape(vector_parts[0]))
                print("vect part 2: ", vector_parts[1][0:10, :], "\n shape of part: ", np.shape(vector_parts[1]))
                print("vect part 3: ", vector_parts[2][0:10, :], "\n shape of part: ", np.shape(vector_parts[2]))
                vector_send = [matrix_merge(vector_parts[0], vector_parts[1], axis=1),
                               matrix_merge(vector_parts[1], vector_parts[2], axis=1),
                               matrix_merge(vector_parts[2], vector_parts[1], axis=1)]
                print("\n shape of vector send: ", np.shape(vector_send))

                # Place the new vector in the que n-s times to be sent to nodes and update counters
                print("count: ", self._count)
                if self._count != self._iterations:
                    for i in range(self._n - self._s):
                        self._sendingQueue.put(vector_send[i])
                        print("Queue size: ", self._sendingQueue.qsize())
                    self._count += 1
                    # Time stamp
                    end_time = time.time()
                    self._time_action_list.append(end_time - start_time)
                    print("Processing iteration time: ", end_time - start_time)
                    self._ready_to_process = False
                else:
                    # When experiment is done:
                    print("Experiment completed")
                    for i in range(self._n - self._s):
                        self._sendingQueue.put("All done")
                    end = time.time()
                    print("runtime (seconds): ", end - start)
                    print("final gradient vector:", b_new[0:15, :], np.shape(b_new))
                    print('Max gradient vector: ', self._max_eigen[0:15, :], np.shape(self._max_eigen))
                    print("\nreceiving list: ", self._time_receive_list)
                    print("\nSending time list: ", self._time_send_list)
                    print("\n Processing list: ", self._time_action_list)
                    print("\n Error List: ", self._error_list)

                    # Do an error check to see how many wrong values we found
                    no_wrong = 0
                    for count, j in enumerate(self._max_eigen):
                        if abs(j - b[count]) >= self._tol:
                            no_wrong += 1
                    print('no_wrong: ', no_wrong)

                    # Save results to a JSON log
                    equalize_list_len(self._time_send_list, self._time_receive_list)
                    equalize_list_len(self._time_send_list, self._time_action_list)
                    equalize_list_len(self._time_send_list, self._error_list)
                    data_dict = {'Send': self._time_send_list, 'Receive': self._time_receive_list,
                                 'Action': self._time_action_list, 'Error': self._error_list}
                    save_to_log(data_dict, self._iterations, self._length)

                    # Plot results
                    # time_plot(self._time_send_list, 'Gradient Coding')
                    # time_plot(self._time_receive_list, 'Gradient Coding')
                    # time_plot(self._time_action_list, 'Gradient Coding')
                    # time_plot(self._error_list, 'Gradient Coding')

                    # End program
                    self._sendingThread.join()
                    self._receivingThread.join()
                    self._actionThread.join()
                    exit()
            else:
                time.sleep(1)


if __name__ == "__main__":
    Agg()

