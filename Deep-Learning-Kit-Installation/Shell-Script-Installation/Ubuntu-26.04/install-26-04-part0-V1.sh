#!/bin/bash

# --------------------------------------------------------------#
# Script to set up a Deep Learning VM on Google Cloud Platform  #
#---------------------------------------------------------------#
# Author: Amir Jafari                                           #
# Date: 06/28/2026                                              #
# Nvidia Driver Version: 580 (open kernel modules)              #
#---------------------------------------------------------------#
# Ubuntu Version: 26.04 (Resolute Raccoon)                      #
# ----------------- Install Nvidia Driver---------------------- #
# NOTE: On 26.04 the recommended driver ships in Ubuntu's own
# 'restricted' repo. We let ubuntu-drivers pick the right branch
# instead of hard-coding a version like the old 24.04 script.
# --------------------------------------------------------------#

# ---- Clean out any previous NVIDIA / CUDA install (safe on a fresh box) ----
sudo apt purge -y 'nvidia-*' 'libnvidia-*' 'cuda-*' '*cudnn*' '*nsight*'
sudo rm -f /etc/apt/sources.list.d/cuda*
sudo rm -rf /usr/local/cuda*
sudo apt autoremove -y && sudo apt autoclean -y

# ---- Update base system + build tools (gcc-15 is the 26.04 default) --------
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential gcc-15 g++-15 dkms \
  linux-headers-$(uname -r) \
  freeglut3-dev libx11-dev libxmu-dev libxi-dev libglu1-mesa-dev \
  ca-certificates wget

# ---- Install the recommended NVIDIA driver -------------------------------- #
# Easiest + most reliable on 26.04: let Ubuntu choose the branch.
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers install
# If you want to pin a specific branch instead, comment the line above and use:
#   sudo apt install -y nvidia-driver-580-open    # Turing/Ampere/Ada/Blackwell
#   sudo apt install -y nvidia-driver-535          # legacy Pascal/Volta cards

sudo reboot

#-------------------------------------------------------#