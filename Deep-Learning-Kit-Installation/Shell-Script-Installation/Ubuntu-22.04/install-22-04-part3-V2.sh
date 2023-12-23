#!/bin/bash

# --------------------------------------------------------------#
# Script to set up a Deep Learning VM on Google Cloud Platform  #
#---------------------------------------------------------------#
#Autor: Amir Jafari		                                          #
#Date: 12/23/2023					                                      #
# ------------------------------------------------------------- #
# -----------------Test cuda 12.1--------------------------------
cat /proc/driver/nvidia/version
nvcc --version
nvidia-smi
# ----------------- Test Cudnn 8.9-------------------------------
sudo apt install libfreeimage3 libfreeimage-dev -y
sudo cp -r /usr/src/cudnn_samples_v8/ $HOME
cd $HOME/cudnn_samples_v8/mnistCUDNN
sudo make
./mnistCUDNN
cd ~
# ----------------- Python 3.10.6  ------------------------------------
sudo apt install -y python3-pip
sudo apt install build-essential libssl-dev libffi-dev python3-dev -y
sudo apt-get install tcl-dev tk-dev python-tk python3-tk -y
sudo pip3 install --upgrade pip

sudo apt install python3-testresources -y
sudo -H pip3 install tensorflow
sudo -H pip3 install -U scikit-learn
sudo -H pip3 install torch torchvision torchaudio

sudo -H pip3 install matplotlib
sudo -H pip3 install pandas
sudo -H pip3 install seaborn

sudo -H pip3 install leveldb

sudo -H pip3 install opencv-python
sudo -H pip3 install pydotplus
sudo -H pip3 install gpustat
sudo -H pip3 install sacred
sudo -H pip3 install pymongo
sudo -H pip3 install openpyxl
sudo -H pip3 install tqdm


sudo -H pip3 install nltk
sudo -H pip3 install pyspellchecker

pip3 install -U pip setuptools wheel
pip3 install -U 'spacy[cuda-autodetect]'
python3 -m spacy download en_core_web_sm

sudo -H pip3 install textacy
sudo -H pip3 install transformers
sudo -H pip3 install datasets

# ----------------- Pycharm 2022 -----------------
wget https://storage.googleapis.com/cuda-deb/pycharm-community-2022.2.tar.gz
sudo tar -zxf pycharm-community-2022.2.tar.gz
sudo ln -s /home/ubuntu/pycharm-community-2022.2/bin/pycharm.sh pycharm

# ----------------- apt get-------------------
sudo apt-get install -y p7zip-full
sudo apt install unzip
sudo apt-get install gedit -y
sudo apt-get install python3-gi-cairo

# ----------------- Chromium Browser-------------------
sudo apt install chromium-browser -y
