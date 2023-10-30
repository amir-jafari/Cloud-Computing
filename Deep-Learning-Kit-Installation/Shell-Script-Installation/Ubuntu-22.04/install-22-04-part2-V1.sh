#!/bin/bash

# --------------------------------------------------------------#
# Script to set up a Deep Learning VM on Google Cloud Platform  #
#---------------------------------------------------------------#
#Autor: Amir Jafari		                                          #
#Date: 07/15/2023					                                      #
# ------------------------------------------------------------- #

# ----------------- Cudnn 8.6 cuda 11.8-------------------------

#wget https://storage.googleapis.com/cuda-deb/libcudnn8_8.2.1.32-1%2Bcuda11.3_amd64.deb
#wget https://storage.googleapis.com/cuda-deb/libcudnn8-dev_8.2.1.32-1%2Bcuda11.3_amd64.deb
#wget https://storage.googleapis.com/cuda-deb/libcudnn8-samples_8.2.1.32-1%2Bcuda11.3_amd64.deb
#
#sudo dpkg -i libcudnn8_8.2.1.32-1+cuda11.3_amd64.deb
#sudo dpkg -i libcudnn8-dev_8.2.1.32-1+cuda11.3_amd64.deb
#sudo dpkg -i libcudnn8-samples_8.2.1.32-1+cuda11.3_amd64.deb

# ----------------- Install Nvidia Driver------------------------
sudo apt install nvidia-driver-525

wget https://storage.googleapis.com/cuda-deb/cudnn-local-repo-ubuntu2204-8.6.0.163_1.0-1_amd64.deb

sudo dpkg -i cudnn-local-repo-ubuntu2204-8.6.0.163_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-8.6.0.163/cudnn-local-FAED14DD-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install libcudnn8=8.6.0.163-1+cuda11.8
sudo apt-get install libcudnn8-dev=8.6.0.163-1+cuda11.8
sudo apt-get install libcudnn8-samples=8.6.0.163-1+cuda11.8

