#!/bin/bash

# --------------------------------------------------------------#
# Script to set up a Deep Learning VM on Google Cloud Platform  #
#---------------------------------------------------------------#
# Author: Amir Jafari                                           #
# Date: 06/28/2026                                              #
# Cuda Version: 13.x (from Ubuntu 26.04 official repos)         #
#---------------------------------------------------------------#
# WHAT CHANGED vs 24.04:
#   Ubuntu 26.04 ships CUDA in its OWN repositories. No more
#   downloading the NVIDIA .deb, no pin file, no keyring copy,
#   no 'ubuntu2204' mismatch. It is now a single apt install.
#
# DO YOU EVEN NEED THIS SCRIPT?
#   - If you ONLY run PyTorch / TensorFlow: NO. Their pip wheels
#     bundle their own CUDA runtime. You can skip parts 1 and 2
#     entirely and just keep the driver (part 0) + frameworks
#     (part 3).
#   - Keep this ONLY if you compile your own CUDA/C++ code with
#     nvcc, or want the cuDNN sample test in part 2.
# --------------------------------------------------------------#

sudo apt-get update

# Full toolkit (nvcc compiler, libraries, headers) from Ubuntu's archive.
sudo apt-get -y install cuda-toolkit

# ---- Put nvcc on PATH (CUDA installs under /usr/local/cuda) ----
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# NOTE: CUDA 13.x targets Turing-class GPUs and newer. If you are on an
# older Pascal/Volta card, install a 12.x toolkit from NVIDIA's repo instead.