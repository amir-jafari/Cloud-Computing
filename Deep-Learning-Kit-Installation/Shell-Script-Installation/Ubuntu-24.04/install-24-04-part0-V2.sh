#!/bin/bash

# --------------------------------------------------------------#
# Script to set up a Deep Learning VM on Google Cloud Platform  #
#---------------------------------------------------------------#
#Autor: Amir Jafari		                                          #
#Date: 12/123/2023				                                      #
#---------------------------------------------------------------#
# ----------------- Install Nvidia Driver------------------------
sudo apt purge nvidia* -y
sudo apt remove nvidia-* -y
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt autoremove -y && sudo apt autoclean -y
sudo rm -rf /usr/local/cuda*

sudo apt update && sudo apt upgrade -y
sudo apt install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev

sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

sudo apt install libnvidia-common-555 libnvidia-gl-555 nvidia-driver-555 -y

sudo reboot
#sudo apt-get install -y nvidia-driver-555-open
#sudo apt-get install -y cuda-drivers-555