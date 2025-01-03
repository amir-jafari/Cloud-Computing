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

# ----------------- Begin Cuda 10.1 -----------------
# Update and Packages
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install git -y
sudo apt-get install linux-headers-$(uname -r)


# ----------------- Install Cuda 10.1 -----------------
wget https://storage.googleapis.com/cuda-deb/cuda-repo-ubuntu1804-10-1-local-10.1.105-418.39_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.105-418.39_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-1-local-10.1.105-418.39/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
sudo apt-get update


sed 1d /etc/environment > /etc/environment
echo 'PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/local/cuda-10.1/bin"' >> /etc/environment

source /etc/environment
sudo reboot



