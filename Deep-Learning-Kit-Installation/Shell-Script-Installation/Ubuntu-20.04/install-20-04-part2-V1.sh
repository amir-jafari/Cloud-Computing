#!/bin/bash

# --------------------------------------------------------------#
# Script to set up a Deep Learning VM on Google Cloud Platform	#
#---------------------------------------------------------------#
#Autor: Amir Jafari		                                    	#
#Date: 08/06/2022					                        #
#                                                               #
# INSTRUCTIONS: When you run this script, make sure you         #
# include the username associated with your instance as         #
# the first parameter. Otherwise, the softwares will not        #
# work properly.   							                    #
# ------------------------------------------------------------- #

# ----------------- Cudnn 8.1.1 cuda 11.2-----------------
wget https://storage.googleapis.com/cuda-deb/libcudnn8_8.1.1.33-1%2Bcuda11.2_amd64.deb
wget https://storage.googleapis.com/cuda-deb/libcudnn8-dev_8.1.1.33-1%2Bcuda11.2_amd64.deb
wget https://storage.googleapis.com/cuda-deb/libcudnn8-samples_8.1.1.33-1%2Bcuda11.2_amd64.deb

sudo dpkg -i libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb
sudo dpkg -i libcudnn8-dev_8.1.1.33-1+cuda11.2_amd64.deb
sudo dpkg -i libcudnn8-samples_8.1.1.33-1+cuda11.2_amd64.deb





