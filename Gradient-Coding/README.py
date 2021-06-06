############################################################################
# This document serves as an introduction to all the Gradient Coding files #
############################################################################

# Created by Henry Crandall as part of Dr Mingyue Ji's Federeated Learning Senior Project 2021

# 1.0 General Introduction
# The goal of the Gradient Coding files is to implement data encoding methods developed by researchers at UT Austin (see uploaded research papers) 
# to mitigate the effect of stragglers. The project created a Federated Learning system out of Raspbery Pies. Three frameworks were created for experimental purposes:
# Gradient Coding, Uncoded, and Straggler. Runtime and error data were gathered for each framework.

# 2.0 Explanation of Files
# Files can be broadly categorized as Agg, Node, or support files. Agg files are meant to be executed on hardware acting as the aggregator.
# Node files are meant to be executed on node hardware. Support files are called by both node and agg objects. 

# 3.0 Local Updates to files
# Some local updates need to be hard-coded into some files. An exhaustive list will not be provided.
# In general, any IP address appearing in scripts should be adjusted to match the local IP address of the hardware being used (i.e. node_object files will need to 
# have the my_ip variable updated to reflect the local Raspberry Pi's IP address). Comments within individual files should help indicate how to make these adjustments.
# Additionally, some local variables can be adjusted to change experiment parameters such as no_iterations. 

# 4.0 Running Files
# To execute the files, type: "python3 FILE_NAME.py" into a command line interface of the device executing the file. For example, to run an experiment to gather 
# data about the gradient encoding framework: 
#    1. On the agg Raspberry Pi execute the agg_object_power_itteration.py file 
#    2. On each node Raspberry Pi execute the node_object_power_itteration.py file
#    3. Wait for experiment completion

# 5.0 Additional Information
# For extra info and help consult the original research papers referenced by this project or request access to the presentations/videos prepared by Henry Crandall
# for the presentation portion of the project.
