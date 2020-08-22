#!/bin/bash

# ----------------- Cudnn 7.5 CUDA 10-----------------
#wget https://storage.googleapis.com/cuda-deb/libcudnn7-dev_7.5.0.56-1%2Bcuda10.0_amd64.deb
#wget https://storage.googleapis.com/cuda-deb/libcudnn7-doc_7.5.0.56-1%2Bcuda10.0_amd64.deb
#wget https://storage.googleapis.com/cuda-deb/libcudnn7_7.5.0.56-1%2Bcuda10.0_amd64.deb


#sudo dpkg -i libcudnn7_7.5.0.56-1+cuda10.0_amd64.deb
#sudo dpkg -i libcudnn7-dev_7.5.0.56-1+cuda10.0_amd64.deb
#sudo dpkg -i libcudnn7-doc_7.5.0.56-1+cuda10.0_amd64.deb

# ----------------- Cudnn 7.6 cuda 10.1-----------------
#wget https://storage.googleapis.com/cuda-deb/libcudnn7_7.6.4.38-1%2Bcuda10.1_amd64.deb
#wget https://storage.googleapis.com/cuda-deb/libcudnn7-doc_7.6.4.38-1%2Bcuda10.1_amd64.deb
#wget https://storage.googleapis.com/cuda-deb/libcudnn7-dev_7.6.4.38-1%2Bcuda10.1_amd64.deb

#sudo dpkg -i libcudnn7_7.6.4.38-1+cuda10.1_amd64.deb
#sudo dpkg -i libcudnn7-dev_7.6.4.38-1+cuda10.1_amd64.deb
#sudo dpkg -i libcudnn7-doc_7.6.4.38-1+cuda10.1_amd64.deb 
# -----------------TEST cuda 10.0-----------------
cat /proc/driver/nvidia/version
nvcc --version
nvidia-smi

# ----------------- Cudnn 7.6 cuda 10.0-----------------
wget https://storage.googleapis.com/cuda-deb/libcudnn7-dev_7.6.4.38-1%2Bcuda10.0_amd64.deb
wget https://storage.googleapis.com/cuda-deb/libcudnn7-doc_7.6.4.38-1%2Bcuda10.0_amd64.deb
wget https://storage.googleapis.com/cuda-deb/libcudnn7_7.6.4.38-1%2Bcuda10.0_amd64.deb
sudo dpkg -i libcudnn7_7.6.4.38-1+cuda10.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.6.4.38-1+cuda10.0_amd64.deb
sudo dpkg -i libcudnn7-doc_7.6.4.38-1+cuda10.0_amd64.deb

# ----------------- TESTC Cudnn 7.6-----------------
sudo cp -r /usr/src/cudnn_samples_v7/ $HOME
cd $HOME/cudnn_samples_v7/mnistCUDNN
sudo make
./mnistCUDNN
cd

# ----------------- Python 3. ------------------------------------
sudo apt install -y python3-pip
sudo apt install build-essential libssl-dev libffi-dev python3-dev -y
sudo apt-get install tcl-dev tk-dev python-tk python3-tk -y
sudo pip3 install --upgrade pip
sudo pip3 install --upgrade setuptools
sudo -H pip3 install matplotlib
sudo -H pip3 install pandas
sudo -H pip3 install h5py
sudo -H pip3 install leveldb
sudo -H pip3 install seaborn
sudo -H pip3 install tensorflow-gpu
sudo -H pip3 install Theano 
sudo -H pip3 install keras
sudo -H pip3 install -U scikit-learn
sudo -H pip3 install cython
sudo -H pip3 install torch
sudo -H pip3 install torchvision
sudo -H pip3 install opencv-python
sudo -H pip3 install lmdb
sudo -H pip3 install sympy
sudo -H pip3 install pydotplus
sudo -H pip3 install gpustat
sudo -H pip3 install xlrd
sudo -H pip3 install sacred
sudo -H pip3 install pymongo
sudo -H pip3 install openpyxl
sudo -H pip3 install tqdm

sudo -H pip3 install nltk
sudo -H pip3 install pyspellchecker
sudo -H pip3 install -U spacy
sudo -H pip3 install textacy
sudo -H pip3 install pymongo
sudo -H pip3 install joblib


sudo python -m spacy download en


# ----------------- Pycharm 2019 -----------------
wget https://storage.googleapis.com/cuda-deb/pycharm-community-2019.1.2.tar.gz
sudo tar -zxf pycharm-community-2019.1.2.tar.gz
sudo ln -s /home/ubuntu/pycharm-community-2019.1.2/bin/pycharm.sh pycharm

# ----------------- apt get-------------------
sudo apt-get install -y stress htop iotop lm-sensors
sudo apt-get install -y python3-skimage
sudo apt-get install -y p7zip-full
sudo apt install unzip
sudo apt-get install gedit -y

# ----------------- Chromium Browser-------------------
sudo apt install chromium-browser -y
