cd installation/
wget https://storage.googleapis.com/cuda-deb/libcudnn8_8.2.1.32-1%2Bcuda11.3_amd64.deb
wget https://storage.googleapis.com/cuda-deb/libcudnn8-dev_8.2.1.32-1%2Bcuda11.3_amd64.deb
wget https://storage.googleapis.com/cuda-deb/libcudnn8-samples_8.2.1.32-1%2Bcuda11.3_amd64.deb

sudo dpkg -i libcudnn8_8.2.1.32-1+cuda11.3_amd64.deb
sudo dpkg -i libcudnn8-dev_8.2.1.32-1+cuda11.3_amd64.deb
sudo dpkg -i libcudnn8-samples_8.2.1.32-1+cuda11.3_amd64.deb
source ~/.bashrc
source /etc/environment
cat /proc/driver/nvidia/version


