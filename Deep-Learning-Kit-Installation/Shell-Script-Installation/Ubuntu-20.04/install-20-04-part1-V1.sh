#!/bin/bash

# --------------------------------------------------------------#
# Script to set up a Deep Learning VM on Google Cloud Platform	#
#---------------------------------------------------------------#
#Autor: Amir Jafari		                                    	#
#Date: 05/26/2019						                        #
#                                                               #
# INSTRUCTIONS: When you run this script, make sure you         #
# include the username associated with your instance as         #
# the first parameter. Otherwise, the softwares will not        #
# work properly.   							                    #
# ------------------------------------------------------------- #

# ----------------- Begin Cuda 11 -----------------
# Update and Packages
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install git -y
sudo apt-get install linux-headers-$(uname -r)


# ----------------- Install Cuda 11 -----------------
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda-repo-ubuntu2004-11-0-local_11.0.2-450.51.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-0-local_11.0.2-450.51.05-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

sed 1d /etc/environment > /etc/environment
echo 'PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/usr/local/cuda-11.0/bin"' >> /etc/environment

source /etc/environment
sudo reboot



