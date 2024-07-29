
lsb_release -a
mkdir installation
cd installation/
wget https://developer.download.nvidia.com/compute/cuda/11.2.2/local_installers/cuda_11.2.2_460.32.03_linux.run
sudo sh cuda_11.2.2_460.32.03_linux.run

gedit ~/.bashrc
	export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
	export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}  
	
source ~/.bashrc

nvidia-smi
nvcc -V

cd installation/
wget https://storage.googleapis.com/cuda-deb/libcudnn8_8.1.1.33-1%2Bcuda11.2_amd64.deb
wget https://storage.googleapis.com/cuda-deb/libcudnn8-dev_8.1.1.33-1%2Bcuda11.2_amd64.deb
wget https://storage.googleapis.com/cuda-deb/libcudnn8-samples_8.1.1.33-1%2Bcuda11.2_amd64.deb
sudo dpkg -i libcudnn8_8.1.1.33-1+cuda11.2_amd64.deb
sudo dpkg -i libcudnn8-dev_8.1.1.33-1+cuda11.2_amd64.deb
sudo dpkg -i libcudnn8-samples_8.1.1.33-1+cuda11.2_amd64.deb
source ~/.bashrc
source /etc/environment
cat /proc/driver/nvidia/version
nvcc --version
nvcc -V

gedit ~/.bashrc
	export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
	export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}
    
source ~/.bashrc
gedit ~/.bashrc

nvidia-smi
nvcc -V
cat /proc/driver/nvidia/version
nvcc --version
nvidia-smi

sudo apt install libfreeimage3 libfreeimage-dev -y
sudo cp -r /usr/src/cudnn_samples_v8/ $HOME
cd $HOME/cudnn_samples_v8/mnistCUDNN
sudo make
./mnistCUDNN
sudo apt install -y python3-pip
sudo apt install build-essential libssl-dev libffi-dev python3-dev -y

cd ~
sudo apt-get install tcl-dev tk-dev python-tk python3-tk -y
sudo pip3 install --upgrade pip
sudo apt install python3-testresources -y
sudo -H pip3 install tensorflow-gpu
sudo -H pip3 install -U scikit-learn
   	
sudo -H pip3 install torch
sudo -H pip3 install torchvision
python3
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
sudo apt-get install -y p7zip-full
sudo apt install unzip

cd installation/
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install ./google-chrome-stable_current_amd64.deb


ip a
sudo apt-get install openssh-server
sudo apt install htop



gsettings set org.gnome.Vino require-encryption false
sudo pip3 install librosa



sudo apt install git
git clone https://github.com/amir-jafari/Deep-Learning.git

while true; do gpustat; sleep 1; done
nvidia-smi
nvcc -V


pip3 list
sudo su-

ip a
nvidia-smi
nvcc -V




python3 -m spacy download en_core_web_sm
pip3 install -U gensim
while true; do gpustat; sleep 1; done


pip3 install Jupyter
ssh -L 8080:localhost:8080 <name>@<ip address>
jupyter notebook --no-browser --port=8080


