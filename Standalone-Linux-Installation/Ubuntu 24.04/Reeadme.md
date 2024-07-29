# 1. OS  and Graphic Installation 
- Install Ubuntu 24.04
  - Download the Ubuntu 22.04 ISO file [LINK HERE](https://ubuntu.com/download/desktop)
- Install Latest Nvidia Graphic Driver
  - Click on Ubuntu Launch pad, Choose additional Driver, Choose the latest Nvidia Driver and Apply Changes
  - Or Install it using terminal command
    - $ sudo ubuntu-drivers install
    - $ sudo apt install nvidia-driver-555

   
# 2. Check CUDA Paths
- Run 2-cuda.sh
- Type the following commands and check the path is right
- $ gedit ~/.bashrc
  - echo 'export PATH=/usr/local/cuda-12.5/bin:$PATH' >> ~/.bashrc
  - echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH' >> ~/.bas
	
- $ source ~/.bashrc
- $ nvidia-smi
- $ nvcc -V
- $ cat /proc/driver/nvidia/version
# 3. Run the following shell scripts
- 3-cudnn.sh
- 4-test_cuda_cudnn.sh
- 5-software_install.sh
- 6-other_softwares.sh

# 4.Other Setting
- For remote Desktop 
  - Click on the setting and sharing section
  - Turn on the remote access and choose user and password
  - Enable VNC
- For File Share
  -  Check Network Drive section of the repo

- For Mobaxterm VNC: 
  - sudo apt install vino
  - gsettings set org.gnome.Vino require-encryption false
  - gsettings set org.gnome.Vino prompt-enabled true