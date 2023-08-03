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


