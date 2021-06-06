# Code to store data to a log file
# HTC --> 03/30/21

import json
import inspect
from datetime import datetime


# Function to save an input dict to a JSON file at the inputted path
def save_to_log(dict_in, num_iter, length, debug=0):
    # Build file-path first
    file_name = inspect.getfile(inspect.currentframe()).split('/')[-1].split('.')[0]
    time = str(datetime.now()).split('.')[0]
    path = 'log/' + file_name + '_' + str(num_iter) + '_' + str(length) + '_' + time
    if debug:
        print('Name of working file: ', file_name)
        print('Current time: ', time)
        print('Built file path: ', path)

    # Save file to file path
    with open(path, 'w') as outfile:
        json.dump(dict_in, outfile)
    return


# A function to help prepare the dictionary input to the save_to_log function
def equalize_list_len(long_list, short_list):
    flag = True
    while flag:
        if len(long_list) > len(short_list):
            short_list.append('NAN')
        else:
            flag = False
    return


'''    
# Test function
dict_example = {'Name': ['Henry', 'Amy', 'JJ'], 'Age': [24, 23, 19],
                'Location': ['Sandy', 'Sandy', 'Grand Island']}
iter = 100
length = 1200
save_to_log(dict_example, iter, length)

'''
