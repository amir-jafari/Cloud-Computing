#!/bin/bash

# --------------------------------------------------------------#
# Script to set up a Deep Learning VM on Google Cloud Platform  #
#---------------------------------------------------------------#
#Autor: Amir Jafari		                                          #
#Date: 12/23/2023					                                      #
# ------------------------------------------------------------- #

# ----------------- Cudnn 8.9 cuda 12.1---------------------
wget https://storage.googleapis.com/cuda-deb/cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb

sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.7.29_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-8.9.7.29/cudnn-local-08A7D361-keyring.gpg /usr/share/keyrings/

sudo apt-get update
sudo apt-get install libcudnn8
sudo apt-get install libcudnn8-dev
sudo apt-get install libcudnn8-samples

