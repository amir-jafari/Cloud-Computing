#!/bin/bash

# --------------------------------------------------------------#
# Script to set up a Deep Learning VM on Google Cloud Platform	#
#---------------------------------------------------------------#
#Autor: Amir Jafari, Michael Arango, Prince Birring				#
#Date: 02/12/2018						                        #
#Organization:  George Washington University                    #
# INSTRUCTIONS: When you run this script, make sure you         #
# include the username associated with your instance as         #
# the first parameter. Otherwise, the softwares will not        #
# work properly.   							                    #
# ------------------------------------------------------------- #
# ----------------- Browser -----------------
sudo apt update
sudo apt upgrade -y

sudo apt install chromium-browser -y
wget https://storage.googleapis.com/cuda-deb/cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb

# ----------------- Cuda -----------------
sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb
sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
sed 1d /etc/environment > /etc/environment
echo 'PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/local/cuda-9.0/bin"' >> /etc/environment
source /etc/environment
nvcc --version
nvidia-smi

# ----------------- Cudnn -----------------
wget https://storage.googleapis.com/cuda-deb/cudnn-9.0-linux-x64-v7.tgz
sudo tar -zxf cudnn-9.0-linux-x64-v7.tgz

cd cuda
sudo cp lib64/* /usr/local/cuda/lib64/
sudo cp include/* /usr/local/cuda/include/

cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
cd ..
# ----------------- Python ------------------------------------
# --- Virtual Environment for Tensorflow - Python 2.7.X -------

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
sudo pip install --upgrade tensorflow-gpu
sudo pip install Theano
sudo pip install keras
#sudo pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl 
sudo pip install torch
sudo pip install torchvision 
sudo pip install numpy --upgrade
sudo pip install -U scikit-learn
 
# deactivate the virtual environment
deactivate

# -- Virtual Environment for Tensorflow - Python 3.5.X -----
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
sudo pip3 install --upgrade tensorflow-gpu
sudo pip3 install pandas --upgrade
sudo pip3 install --upgrade numexpr
sudo pip3 install --upgrade numpy
sudo pip3 install Theano 
sudo pip3 install keras
sudo pip3 install protobuf
sudo pip3 install sklearn
sudo pip3 install cython
#sudo pip3 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl 
sudo pip3 install torch 
sudo pip3 install torchvision
sudo pip3 install -U scikit-learn
# deactivate the virtual environment
deactivate

sudo pip install --upgrade pip
sudo apt-get install -y p7zip-full
sudo apt install unzip

# ----------------- Pycharm -----------------
# wget https://storage.googleapis.com/cuda-deb/pycharm-community_2016.3-mm1_all.deb
# sudo dpkg -i pycharm-community_2016.3-mm1_all.deb

wget https://storage.googleapis.com/cuda-deb/pycharm-community_2017.3.4-1_amd64.deb
sudo dpkg -i pycharm-community_2017.3.4-1_amd64.deb
# ----------------- Torch -----------------
sudo apt install git
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch 
export TORCH_NVCC_FLAGS="-D__CUDA_NO_HALF_OPERATORS__"
bash install-deps

./install.sh
sudo apt update
cd ..
source ~/.bashrc

sudo apt-get install -y luarocks
sudo ~/torch/install/bin/luarocks install torch 
sudo ~/torch/install/bin/luarocks install image 
sudo ~/torch/install/bin/luarocks install nngraph
sudo ~/torch/install/bin/luarocks install optim
sudo ~/torch/install/bin/luarocks install nn
sudo ~/torch/install/bin/luarocks install cutorch
sudo ~/torch/install/bin/luarocks install cunn
sudo ~/torch/install/bin/luarocks install cunnx
sudo ~/torch/install/bin/luarocks install dp

sudo apt-get install --no-install-recommends libhdf5-serial-dev liblmdb-dev

sudo ~/torch/install/bin/luarocks install tds
sudo ~/torch/install/bin/luarocks install "https://raw.github.com/deepmind/torch-hdf5/master/hdf5-0-0.rockspec"
sudo ~/torch/install/bin/luarocks install "https://raw.github.com/Neopallium/lua-pb/master/lua-pb-scm-0.rockspec"
sudo ~/torch/install/bin/luarocks install lightningmdb 0.9.18.1-1 LMDB_INCDIR=/usr/include LMDB_LIBDIR=/usr/lib/x86_64-linux-gnu

sudo ~/torch/install/bin/luarocks install "httpsraw.githubusercontent.comngimelnccl.torchmasternccl-scm-1.rockspec"


git clone https://github.com/torch/demos
sudo apt-get install gnuplot-x11

sudo apt update
# ----------------- ZeroBrane -----------------
wget https://storage.googleapis.com/cuda-deb/ZeroBraneStudioEduPack-1.60-linux.sh

chmod +x ZeroBraneStudioEduPack-1.60-linux.sh
./ZeroBraneStudioEduPack-1.60-linux.sh
#----------------- Caffe -----------------
sudo apt-get upgrade -y

sudo apt-get install -y opencl-headers build-essential protobuf-compiler \
    libprotoc-dev libboost-all-dev libleveldb-dev hdf5-tools libhdf5-serial-dev \
    libopencv-core-dev  libopencv-highgui-dev libsnappy-dev \
    libatlas-base-dev cmake libstdc++6-4.8-dbg libgoogle-glog0v5 libgoogle-glog-dev \
    libgflags-dev liblmdb-dev git python-pip gfortran libopencv-dev

sudo apt-get clean

sudo apt-get install -y linux-image-extra-`uname -r` linux-headers-`uname -r` linux-image-`uname -r`

sudo apt-get clean

sudo sh -c "sudo echo '/usr/local/cuda/lib64' > /etc/ld.so.conf.d/cuda_hack.conf"
sudo ldconfig /usr/local/cuda/lib64


git clone https://github.com/BVLC/caffe.git
cd caffe
cd python
for req in $(cat requirements.txt); do sudo pip install $req; done

cd ../
cp Makefile.config.example Makefile.config

sed -i '/^# USE_CUDNN := 1/s/^# //' Makefile.config

sed -i '/^# WITH_PYTHON_LAYER := 1/s/^# //' Makefile.config
sed -i 's/\/usr\/local\/cuda/\/usr\/local\/cuda-9.0/g' Makefile.config
sed -i 's/\/usr\/local\/include/\/usr\/local\/include \/usr\/include\/hdf5\/serial/g' Makefile.config
sed -i '/^PYTHON_INCLUDE/a    /usr/local/lib/python2.7/dist-packages/numpy/core/include/ \\' Makefile.config

sed -i '/-gencode arch=compute_20,code=sm_20/s/^/#/g' Makefile.config
sed -i '/-gencode arch=compute_20,code=sm_21/s/^/#/g' Makefile.config
sed -i '/-gencode arch=compute_30,code=sm_30/s/^/CUDA_ARCH :=  /g' Makefile.config

sudo sed -i '/LIBRARY_DIRS/s/^/#/g' Makefile.config
sudo sed -i '/LIBRARY_DIRS/a LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/hdf5/serial' Makefile.config

sudo ln -s libhdf5_serial.so.10.1.0 libhdf5.so
sudo ln -s libhdf5_serial_hl.so.10.0.2 libhdf5_hl.so

# And finally build!
make -j 8 all py

make -j 8 test
make runtest

echo "export PYTHONPATH=/opt/cat-dogs/repo/caffe/python:$PYTHONPATH" >> ~/.bashrc


echo "export PYTHONPATH=/home/$1/caffe/python" >> ~/.bashrc

source ~/.bashrc
sudo ln /dev/null /dev/raw1394
sudo apt-get -y install python-skimage
sudo apt-get -y install python-pydot
sudo apt-get -y install python-protobuf 
sudo rm -rf /dev/raw1394

cd ..
 ----------------- Caffe2 -----------------
#sudo apt-get install -y --no-install-recommends libgflags-dev
#sudo apt-get install -y --no-install-recommends \
#      libgtest-dev \
#      libiomp-dev \
#      libleveldb-dev \
#      liblmdb-dev \
#      libopencv-dev \
#      libopenmpi-dev \
#      libsnappy-dev \
#      openmpi-bin \
#      openmpi-doc \
#      python-pydot
#
#source ~/python2/bin/activate
#
#sudo python -m pip install \
#      flask \
#      future \
#      graphviz \
#      hypothesis \
#      jupyter \
#      matplotlib \
#      pydot python-nvd3 \
#      pyyaml \
#      requests \
#      scikit-image \
#      scipy \
#      setuptools \
#      six \
#      tornado
#deactivate
#
#git clone --recursive https://github.com/caffe2/caffe2.git && cd caffe2
#make && cd build && sudo make install
#python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
#
# Export the path to .bashrc and source it 
echo "export PYTHONPATH=/home/$1/caffe/python:/home/$1/caffe2/build" >> ~/.bashrc
source ~/.bashrc
source /etc/environment
#python -m caffe2.python.operator_test.relu_op_test
#
cd ~
#------------------Forget---------------------------
sudo pip2 install opencv-python
sudo pip2 install lmdb
sudo pip2 install sympy
sudo pip install sympy
sudo pip install pydotplus

