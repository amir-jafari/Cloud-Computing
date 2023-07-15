#!/bin/bash

# --------------------------------------------------------------#
# Script to set up a Deep Learning VM on Google Cloud Platform  #
#---------------------------------------------------------------#
#Autor: Amir Jafari		                                  #
#Date: 08/06/2022					                      #
# ------------------------------------------------------------- #

# ----------------- Cudnn 8.2 cuda 11.6-------------------------

wget https://storage.googleapis.com/cuda-deb/libcudnn8_8.2.1.32-1%2Bcuda11.3_amd64.deb
wget https://storage.googleapis.com/cuda-deb/libcudnn8-dev_8.2.1.32-1%2Bcuda11.3_amd64.deb
wget https://storage.googleapis.com/cuda-deb/libcudnn8-samples_8.2.1.32-1%2Bcuda11.3_amd64.deb

sudo dpkg -i libcudnn8_8.2.1.32-1+cuda11.3_amd64.deb
sudo dpkg -i libcudnn8-dev_8.2.1.32-1+cuda11.3_amd64.deb
sudo dpkg -i libcudnn8-samples_8.2.1.32-1+cuda11.3_amd64.deb



