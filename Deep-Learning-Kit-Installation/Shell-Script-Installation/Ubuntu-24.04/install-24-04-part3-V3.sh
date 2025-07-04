#!/bin/bash

# --------------------------------------------------------------#
# Script to set up a Deep Learning VM on Google Cloud Platform  #
#---------------------------------------------------------------#
#Autor: Amir Jafari		                                          #
#Date: 07/04/2025					                                      #
# ------------------------------------------------------------- #
# -----------------Test cuda 12.5--------------------------------
cat /proc/driver/nvidia/version
nvcc --version
nvidia-smi
# ----------------- Test Cudnn 9.5.1 -------------------------------
sudo apt install libfreeimage3 libfreeimage-dev -y
sudo cp -r /usr/src/cudnn_samples_v9/ $HOME
cd $HOME/cudnn_samples_v9/mnistCUDNN
sudo make
./mnistCUDNN
cd ~
# ----------------- Python 3.12  ------------------------------------
sudo apt install -y python3-pip
sudo rm /usr/lib/python3.12/EXTERNALLY-MANAGED
pip3 install -U scikit-learn
pip3 install torch==2.7.1
pip3 install torchvision==0.22.1
pip3 install torchaudio==2.7.1
pip3 install tensorflow[and-cuda]==2.19.0
pip3 install --upgrade \
  nvidia-cublas-cu12==12.6.4.1 \
  nvidia-cuda-cupti-cu12==12.6.80 \
  nvidia-cuda-nvrtc-cu12==12.6.77 \
  nvidia-cuda-runtime-cu12==12.6.77 \
  nvidia-cudnn-cu12==9.5.1.17 \
  nvidia-cufft-cu12==11.3.0.4 \
  nvidia-curand-cu12==10.3.7.77 \
  nvidia-cusolver-cu12==11.7.1.2 \
  nvidia-cusparse-cu12==12.5.4.2 \
  nvidia-nccl-cu12==2.26.2 \
  nvidia-nvjitlink-cu12==12.6.85

pip3 install matplotlib
pip3 install pandas
pip3 install seaborn
pip3 install opencv-python
pip3 install pydotplus
pip3 install gpustat
pip3 install sacred
pip3 install openpyxl
pip3 install tqdm
pip3 install nltk
pip3 install pyspellchecker
pip3 install -U 'spacy[cuda12x]'
python3 -m spacy download en_core_web_sm
pip3 install textacy
pip3 install transformers
pip3 install datasets
pip3 install librosa
pip3 install Jupyter
pip3 install tf-keras
pip3 install torchinfo
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
