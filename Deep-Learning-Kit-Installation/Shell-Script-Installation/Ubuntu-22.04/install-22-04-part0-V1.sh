#!/bin/bash

# --------------------------------------------------------------#
# Script to set up a Deep Learning VM on Google Cloud Platform  #
#---------------------------------------------------------------#
#Autor: Amir Jafari		                                          #
#Date: 07/15/2023				                                        #
#---------------------------------------------------------------#
# ----------------- Install Nvidia Driver------------------------
### If you have previous installation remove it first.
sudo apt purge nvidia* -y
sudo apt remove nvidia-* -y
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt autoremove -y && sudo apt autoclean -y
sudo rm -rf /usr/local/cuda*

# system update
sudo apt update && sudo apt upgrade -y

# install other import packages
sudo apt install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev

# first get the PPA repository driver
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update


# install nvidia driver with dependencies
sudo apt install libnvidia-common-515 libnvidia-gl-515 nvidia-driver-515 -y

# reboot
sudo reboot
