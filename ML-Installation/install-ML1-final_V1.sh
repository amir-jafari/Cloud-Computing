#!/bin/bash

# ---------------------------------------------------------------------------------- #
# 			Script to set up a Deep Learning VM on Google Cloud Platform			 #
#           ------------------------------------------------------------			 #
#Autor:             Amir Jafari, Michael Arango, Prince Birring						 #
#Date:              09/23/2017						                                 #
#Organization:      George Washington University                                     #
# INSTRUCTIONS: When you run this script, make sure you include the username 		 #
# 				associated with your instance as the first parameter. Otherwise,	 #
# 				the softwares will not work properly.   							 #
# ---------------------------------------------------------------------------------- #

# ------------------------ Cuda Installation ------------------------

# Update packages
sudo apt update
# Instal chromium browser
sudo apt install -y chromium-browser
sudo apt-get update



# -------------------- Python Required Package Installations --------------------

# ----------------- Python ----------------------------------------------------
# ------------- Virtual Environment for Tensorflow - Python 2.7.X -------------

# Install libraries needed to make a virtual environment
sudo apt-get install -y python-pip python-dev python-virtualenv
# Create a virtual environment, tensorflow2
virtualenv --system-site-packages python2
# Activate the virtual environment 
source ~/python2/bin/activate
# Install pip in virtual environment
easy_install -U pip
# Make sure the tensorflow package is up-to-date
sudo apt-get install -y python-pip python-dev
sudo apt-get install -y python-tk
sudo apt-get install -y python-matplotlib
sudo apt-get install -y python-pandas
#sudo apt-get install -y python-sklearn
sudo apt-get install -y python-skimage
sudo apt-get install -y python-h5py
sudo apt-get install -y python-leveldb
sudo apt-get install -y python-protobuf
sudo apt-get install -y python-gflags
sudo apt-get install -y python-seaborn
sudo apt-get install -y python-networkx
sudo pip install --upgrade pip
sudo pip install numpy --upgrade
sudo pip install pydot
sudo pip install pydotplus
sudo pip install sympy
sudo pip install -U scikit-learn
 
# deactivate the virtual environment
deactivate

# ------------- Virtual Environment for Tensorflow - Python 3.5.X -------------
# Install libraries needed to make a virtual environment
sudo apt-get install -y python3-pip python3-dev python-virtualenv
# Create a virtual environment, tensorflow3
virtualenv --system-site-packages -p python3 python3
# Activate the virtual environment 
source ~/python3/bin/activate
# Install pip in virtual environment
easy_install -U pip
# Make sure the tensorflow package is up-to-date
sudo apt-get install -y python3-pip python3-dev
sudo apt-get install -y python3-tk
sudo apt-get install -y python3-pip
sudo apt-get install -y python3-matplotlib
sudo apt-get install -y python3-numpy
sudo apt-get install -y python3-pandas
sudo apt-get install -y python3-skimage
sudo apt-get install -y python3-h5py
sudo apt-get install -y python3-leveldb
sudo apt-get install -y python3-yaml
sudo apt-get install -y python3-networkx
sudo apt-get install -y python3-seaborn
sudo pip3 install --upgrade pip
sudo pip3 install pandas --upgrade
sudo pip3 install --upgrade numexpr
sudo pip3 install --upgrade numpy
sudo pip3 install protobuf
#sudo pip3 install sklearn
sudo pip3 install -U scikit-learn
sudo pip3 install cython
sudo pip3 install pydot
sudo pip3 install pydotplus
sudo pip3 install graphviz
sudo pip3 install sympy
sudo pip3 install opencv-python
# deactivate the virtual environment
deactivate

# install packages to zip and unzip files 
sudo apt-get install -y p7zip-full
sudo apt install unzip

# ------------------------ Pycharm Installation ------------------------

# Download .deb file from google storage bucket
wget https://storage.googleapis.com/cuda-deb/pycharm-community_2016.3-mm1_all.deb
# unpack the contents
sudo dpkg -i pycharm-community_2016.3-mm1_all.deb

# ------------------------ R and Rstudio Installation ------------------------
sudo add-apt-repository 'deb https://ftp.ussg.iu.edu/CRAN/bin/linux/ubuntu trusty/'
sudo apt-get update
sudo apt-get install r-base
sudo apt-get install r-base-dev

sudo apt-get install gdebi-core
wget https://download1.rstudio.org/rstudio-1.0.44-amd64.deb
sudo gdebi rstudio-1.0.44-amd64.deb




sudo apt-get install graphviz libgraphviz-dev
sudo apt-get install pandoc

sudo pip2 install opencv-python
sudo pip2 install lmdb













