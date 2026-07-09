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
pip install --upgrade pip==26.1.2 setuptools==70.2.0 wheel==0.47.0

# ----------------- PyTorch (CUDA 13.0 build) ------------------ #
# 'pip install torch' on PyPI already defaults to the CUDA 13.0
# wheel on Linux, but we pin the index explicitly to be safe.
# For Pascal/Volta GPUs use cu126 instead of cu130 below.
pip install torch==2.12.1 torchvision==0.27.1 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130

# ----------------- TensorFlow --------------------------------- #
# [and-cuda] pulls the CUDA libs TF needs; safe to run alongside torch.
pip install tensorflow[and-cuda]==2.21.0
pip install tf-keras==2.21.0

# ----------------- Core scientific / DL stack ---------------- #
pip install -U scikit-learn==1.9.0
pip install matplotlib==3.11.0
pip install pandas==3.0.4
pip install seaborn==0.13.2
pip install opencv-python==4.13.0.92
pip install pydotplus==2.0.2
pip install gpustat==1.1.1
pip install sacred==0.8.7
pip install openpyxl==3.1.5
pip install tqdm==4.68.3
pip install nltk==3.9.4
pip install pyspellchecker==0.9.0
# NOTE: do NOT use spacy[cuda12x] on Py3.13 -- it forces an old spaCy that
# fails to compile (blis 0.7 vs modern numpy). Plain spaCy already uses the
# GPU through PyTorch, so the extra is unnecessary anyway.
pip install -U spacy==3.8.14
python -m spacy download en_core_web_sm==3.8.0
# textacy removed: it is unmaintained and pulls 'floret', which fails to build
# on GCC 15 (missing <cstdint>). Skip unless you specifically need it.
pip install transformers==5.12.1
pip install datasets==5.0.0
pip install librosa==0.11.0
pip install jupyter==1.1.1
pip install torchinfo==1.8.0

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