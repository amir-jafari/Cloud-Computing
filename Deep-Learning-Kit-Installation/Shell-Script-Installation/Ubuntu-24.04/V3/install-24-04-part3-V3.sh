#!/bin/bash

# --------------------------------------------------------------#
# Script to set up a Deep Learning VM on Google Cloud Platform  #
#---------------------------------------------------------------#
#Autor: Amir Jafari		                                          #
#Date: 07/04/2025					                                      #
# ------------------------------------------------------------- #
# -----------------Test cuda 12.6.2--------------------------------
cat /proc/driver/nvidia/version
nvcc --version
nvidia-smi
# ----------------- Test Cudnn 9.6.0 -------------------------------
sudo apt install libfreeimage3 libfreeimage-dev -y
sudo cp -r /usr/src/cudnn_samples_v9/ $HOME
cd $HOME/cudnn_samples_v9/mnistCUDNN
sudo make
./mnistCUDNN
cd ~
# ----------------- Python 3.12  ------------------------------------
sudo apt install -y python3-pip python3-dev
sudo rm /usr/lib/python3.12/EXTERNALLY-MANAGED

# ----------------- Deep Learning Frameworks ------------------------------------
# PyTorch 2.5.1 with CUDA 12.6 support
pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# TensorFlow 2.18.0 with CUDA 12.x support
pip3 install tensorflow==2.18.0

# TensorFlow CUDA dependencies for CUDA 12.6
pip3 install --upgrade \
  nvidia-cublas-cu12==12.6.4.1 \
  nvidia-cuda-cupti-cu12==12.6.80 \
  nvidia-cuda-nvrtc-cu12==12.6.77 \
  nvidia-cuda-runtime-cu12==12.6.77 \
  nvidia-cudnn-cu12==9.6.0.74 \
  nvidia-cufft-cu12==11.3.0.4 \
  nvidia-curand-cu12==10.3.7.77 \
  nvidia-cusolver-cu12==11.7.1.2 \
  nvidia-cusparse-cu12==12.5.4.2 \
  nvidia-nccl-cu12==2.26.2 \
  nvidia-nvjitlink-cu12==12.6.85

# TensorFlow Keras
pip3 install tf-keras==2.18.0

# ----------------- Scientific Computing & ML Libraries ------------------------------------
pip3 install scikit-learn==1.6.0
pip3 install numpy==1.26.4
pip3 install scipy==1.14.1
pip3 install pandas==2.2.3
pip3 install matplotlib==3.10.0
pip3 install seaborn==0.13.2
pip3 install opencv-python==4.10.0.84
pip3 install Pillow==11.1.0

# ----------------- NLP & Transformers Libraries ------------------------------------
# Hugging Face ecosystem - using stable versions before v5 breaking changes
pip3 install transformers==4.47.1
pip3 install tokenizers==0.21.0
pip3 install datasets==3.2.0
pip3 install accelerate==1.2.1
pip3 install sentencepiece==0.2.0
pip3 install protobuf==5.29.2

# NLP tools
pip3 install nltk==3.9.1
pip3 install spacy==3.8.3
python3 -m spacy download en_core_web_sm
pip3 install textacy==0.13.0
pip3 install pyspellchecker==0.8.2

# ----------------- Audio Processing ------------------------------------
pip3 install librosa==0.10.2.post1
pip3 install soundfile==0.12.1

# ----------------- Development & Utilities ------------------------------------
pip3 install jupyter==1.1.1
pip3 install ipython==8.31.0
pip3 install notebook==7.3.2
pip3 install jupyterlab==4.3.4
pip3 install torchinfo==1.8.0
pip3 install tqdm==4.67.1
pip3 install gpustat==1.1.1
pip3 install pydot==3.0.4
pip3 install pydotplus==2.0.2
pip3 install openpyxl==3.1.5
pip3 install sacred==0.8.5
pip3 install h5py==3.12.1
pip3 install tensorboard==2.18.0
# ----------------- Pycharm -----------------
sudo apt-get install openjdk-11-jre
sudo snap install pycharm-community --classic

# ----------------- apt get-------------------
sudo apt-get install -y p7zip-full
sudo apt install unzip
sudo apt-get install gedit -y
sudo apt  install nvtop
sudo apt-get install openssh-server
sudo apt install htop
# ----------------- google Browser-------------------
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install ./google-chrome-stable_current_amd64.deb
