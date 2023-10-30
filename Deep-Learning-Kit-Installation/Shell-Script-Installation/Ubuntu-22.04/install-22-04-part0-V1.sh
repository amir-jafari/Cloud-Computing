#!/bin/bash

# --------------------------------------------------------------#
# Script to set up a Deep Learning VM on Google Cloud Platform  #
#---------------------------------------------------------------#
#Autor: Amir Jafari		                                          #
#Date: 07/15/2023				                                        #
#---------------------------------------------------------------#
# ----------------- Install Nvidia Driver------------------------
#sudo apt-get purge nvidia-*
#sudo apt-get update
#sudo apt-get autoremove
#sudo apt install nvidia-driver-520
sudo apt update && sudo apt upgrade
sudo apt autoremove nvidia* --purge
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo apt install nvidia-driver-525
sudo reboot