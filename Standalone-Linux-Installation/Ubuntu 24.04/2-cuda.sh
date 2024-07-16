sudo apt update && sudo apt upgrade -y
sudo apt install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev


wget https://developer.download.nvidia.com/compute/cuda/12.5.1/local_installers/cuda_12.5.1_555.42.06_linux.run
sudo sh cuda_12.5.1_555.42.06_linux.run

# Add export to bashrc
# nano ~/.bashrc
export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PATH=/usr/local/cuda-12.5/bin${PATH:+:${PATH}}
source ~/.bashrc
nvidia-smi

