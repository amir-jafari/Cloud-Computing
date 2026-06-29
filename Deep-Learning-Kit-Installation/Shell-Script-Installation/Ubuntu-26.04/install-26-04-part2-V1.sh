#!/bin/bash

# --------------------------------------------------------------#
# Script to set up a Deep Learning VM on Google Cloud Platform  #
#---------------------------------------------------------------#
# Author: Amir Jafari                                           #
# Date: 06/28/2026                                              #
# Cudnn Version: 9.x (from Ubuntu 26.04 official repos)         #
# ------------------------------------------------------------- #
# WHAT CHANGED vs 24.04:
#   No more downloading the cuDNN local .deb installer and
#   copying keyrings. cuDNN is now a plain apt package on 26.04.
#
# DO YOU NEED THIS?
#   Same as part 1: ONLY if you compile against cuDNN directly
#   or want the mnistCUDNN sample test in part 3. PyTorch/TF
#   bundle their own cuDNN, so framework-only users can skip it.
# ------------------------------------------------------------- #

sudo add-apt-repository -y multiverse && sudo apt update
sudo apt install -y nvidia-cudnn
# If you specifically need the CUDA 13 build: sudo apt-get -y install cudnn-cuda-13