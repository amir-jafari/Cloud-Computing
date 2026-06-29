#!/bin/bash

# --------------------------------------------------------------#
# Script to set up a Deep Learning VM on Google Cloud Platform  #
#---------------------------------------------------------------#
# Author: Amir Jafari                                           #
# Date: 06/28/2026                                              #
# ------------------------------------------------------------- #
# THE IMPORTANT PART. Read this before running.
#
# Ubuntu 26.04 ships Python 3.14 as the system interpreter.
# As of now:
#   - TensorFlow (2.21) only has wheels for Python 3.10 - 3.13.
#   - PyTorch CUDA wheels on 3.14 are unreliable (often CPU-only).
#   - 26.04 also blocks 'pip install' into the system Python
#     (externally-managed-environment).
# So we DO NOT pip into the system Python like the old 24.04
# script did. Instead we install Python 3.13 from the Deadsnakes
# PPA and build a clean virtual environment for the ML stack.
# Activate it with:  source ~/dl-venv/bin/activate
# ------------------------------------------------------------- #

# ----------------- Test the driver / CUDA ---------------------
cat /proc/driver/nvidia/version
nvcc --version          # (only works if you installed part 1)
nvidia-smi

# ----------------- Test cuDNN (needs parts 1 & 2) -------------
# Skip this block if you went framework-only.
sudo apt install -y libfreeimage3 libfreeimage-dev
if [ -d /usr/src/cudnn_samples_v9 ]; then
  cp -r /usr/src/cudnn_samples_v9/ "$HOME"
  cd "$HOME/cudnn_samples_v9/mnistCUDNN" && make && ./mnistCUDNN
  cd ~
fi

# ----------------- Python 3.13 + venv ------------------------- #
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.13 python3.13-venv python3.13-dev

# Build an isolated environment for all ML packages
python3.13 -m venv ~/dl-venv
source ~/dl-venv/bin/activate
pip install --upgrade pip setuptools wheel

# ----------------- PyTorch (CUDA 13.0 build) ------------------ #
# 'pip install torch' on PyPI already defaults to the CUDA 13.0
# wheel on Linux, but we pin the index explicitly to be safe.
# For Pascal/Volta GPUs use cu126 instead of cu130 below.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# ----------------- TensorFlow --------------------------------- #
# [and-cuda] pulls the CUDA libs TF needs; safe to run alongside torch.
pip install tensorflow[and-cuda]
pip install tf-keras

# ----------------- Core scientific / DL stack ---------------- #
pip install -U scikit-learn
pip install matplotlib
pip install pandas
pip install seaborn
pip install opencv-python
pip install pydotplus
pip install gpustat
pip install sacred
pip install openpyxl
pip install tqdm
pip install nltk
pip install pyspellchecker
pip install -U 'spacy[cuda12x]'
python -m spacy download en_core_web_sm
pip install textacy
pip install transformers
pip install datasets
pip install librosa
pip install jupyter
pip install torchinfo

# ----------------- Quick GPU sanity check --------------------- #
python - <<'PY'
import torch, tensorflow as tf
print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("tf:", tf.__version__, "| gpus:", tf.config.list_physical_devices('GPU'))
PY
deactivate

# ----------------- Pycharm ----------------------------------- #
sudo apt-get install -y openjdk-25-jre
sudo snap install pycharm-community --classic

# ----------------- apt-get utilities ------------------------- #
sudo apt-get install -y p7zip-full
sudo apt install -y unzip
sudo apt-get install -y gedit
sudo apt install -y nvtop
sudo apt-get install -y openssh-server
sudo apt install -y htop

# ----------------- Google Chrome ----------------------------- #
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install -y ./google-chrome-stable_current_amd64.deb