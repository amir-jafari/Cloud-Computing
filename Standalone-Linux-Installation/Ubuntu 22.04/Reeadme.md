# 1. OS  and Graphic Installation 
- Install Ubuntu 22.04
  - Download the Ubuntu 22.04 ISO file [LINK HERE](https://ubuntu.com/download/desktop/thank-you?version=22.04.2&architecture=amd64)
- Install Latest Nvidia Graphic Driver
  - click on Ubuntu Launch pad
  - Choose additional Driver
  - Choose the latest Nvidia Driver and Apply Changes
   
# 2. Check CUDA Paths
- Run 2-cuda.sh
- Type the following commands and check the path is right
- $ gedit ~/.bashrc
  - export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
  - export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}  
	
- $ source ~/.bashrc
- $ nvidia-smi
- $ nvcc -V
- $ cat /proc/driver/nvidia/version
- 
# 3. Run the following shell scripts
- 3-cudnn.sh
- 4-test_cuda_cudnn.sh
- 5-software_install.sh
- 6-other_softwares.sh

# 4. GWU VNC Setting
- gsettings set org.gnome.Vino require-encryption false