#!/bin/bash

# --------------------------------------------------------------#
# Script to set up a Deep Learning VM on Google Cloud Platform  #
#---------------------------------------------------------------#
#Autor: Amir Jafari		                                          #
#Date: 12/20/2024					                                      #
# ------------------------------------------------------------- #
# -----------------Test cuda 12.5--------------------------------
cat /proc/driver/nvidia/version
nvcc --version
nvidia-smi
# ----------------- Test Cudnn 8.9-------------------------------
sudo apt install libfreeimage3 libfreeimage-dev -y
sudo cp -r /usr/src/cudnn_samples_v9/ $HOME
cd $HOME/cudnn_samples_v9/mnistCUDNN
sudo make
./mnistCUDNN
cd ~
# ----------------- Python 3.10.6  ------------------------------------
sudo apt install -y python3-pip
sudo rm /usr/lib/python3.12/EXTERNALLY-MANAGED
#sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev pkg-config wget
#sudo apt install python3-testresources -y
pip3 install tensorflow[and-cuda]==2.17.0
pip3 install -U scikit-learn
pip3 install torch==2.3.1
pip3 install torchvision==0.18.1
pip3 install torchaudio==2.3.1
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
pip3 install -U 'spacy[cuda11x]'
python3 -m spacy download en_core_web_sm
pip3 install textacy
pip3 install transformers
pip3 install datasets
pip3 install librosa
pip3 install Jupyter

# ----------------- Pycharm 2024 -----------------
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
