#!/bin/bash

# --------------------------------------------------------------#
# Script to set up a Deep Learning VM on Google Cloud Platform  #
#---------------------------------------------------------------#
#Autor: Amir Jafari		                                  #
#Date: 08/06/2022					                #
#---------------------------------------------------------------#



# -----------------TEST cuda 11.6--------------------------------
cat /proc/driver/nvidia/version
nvcc --version
nvidia-smi


# ----------------- TEST Cudnn 8.2-------------------------------
sudo apt install libfreeimage3 libfreeimage-dev -y
sudo cp -r /usr/src/cudnn_samples_v8/ $HOME
cd $HOME/cudnn_samples_v8/mnistCUDNN
sudo make
./mnistCUDNN
cd ~


# ----------------- Python 3.8 ------------------------------------
sudo apt install -y python3-pip
sudo apt install build-essential libssl-dev libffi-dev python3-dev -y
sudo apt-get install tcl-dev tk-dev python-tk python3-tk -y
sudo pip3 install --upgrade pip

sudo apt install python3-testresources -y
sudo -H pip3 install tensorflow-gpu
sudo -H pip3 install -U scikit-learn
sudo -H pip3 install torch
sudo -H pip3 install torchvision

sudo -H pip3 install matplotlib
sudo -H pip3 install pandas
sudo -H pip3 install seaborn
sudo -H pip3 install h5py
sudo -H pip3 install leveldb

sudo -H pip3 install opencv-python
sudo -H pip3 install sympy
sudo -H pip3 install pydotplus
sudo -H pip3 install gpustat
sudo -H pip3 install sacred
sudo -H pip3 install pymongo
sudo -H pip3 install openpyxl
sudo -H pip3 install tqdm


sudo -H pip3 install nltk
sudo -H pip3 install pyspellchecker
sudo -H pip3 install -U spacy
sudo python3 -m spacy download en
sudo -H pip3 install textacy
sudo -H pip3 install transformers
sudo -H pip3 install datasets
sudo -H pip3 install torchtext




# ----------------- Pycharm 2022 -----------------
wget https://storage.googleapis.com/cuda-deb/pycharm-community-2022.2.tar.gz
sudo tar -zxf pycharm-community-2020.3.2.tar.gz
sudo ln -s /home/ubuntu/pycharm-community-2020.3.2/bin/pycharm.sh pycharm

# ----------------- apt get-------------------
sudo apt-get install -y p7zip-full
sudo apt install unzip
sudo apt-get install gedit -y
sudo apt-get install python3-gi-cairo

# ----------------- Chromium Browser-------------------
sudo apt install chromium-browser -y
