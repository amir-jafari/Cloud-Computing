# Ubuntu 24.04 Deep Learning Software Shell Script GUIDE

## Getting Started - Option 1
Latest Version: To install all the Frameworks (tensorflow, pytorch), launch your VM  and run the following commands in order in to your VM terminal 
Cuda 12.5 Cudnn 9.2

```
sudo apt install git -y
```
```
git clone https://github.com/amir-jafari/Cloud-Computing.git
```
```
cd Cloud-Computing/Deep-Learning-Kit-Installation/Shell-Script-Installation/Ubuntu-24.04/
```
We are going to install:
```
mv install-24-04-part0-V2.sh ~
```
```
mv install-24-04-part1-V2.sh ~
```
```
mv install-24-04-part2-V2.sh ~
```
```
mv install-24-04-part3-V2.sh ~
```
```
cd ~
```
```
chmod +x install-24-04-part0-V2.sh
```
```
sudo ./install-24-04-part0-V2.sh
```
By this time you should have Nvidia installed correctly. 
```
chmod +x install-24-04-part1-V2.sh
```
```
sudo ./install-24-04-part1-V2.sh
```
By this time you should have CUDA 12.5 installed correctly. 
```
nano ~/.bashrc
```

add following lines to the end of the script
```
echo 'export PATH=/usr/local/cuda-12.5/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```
do CTRL+X then press y key and then ENTER.

```
source ~/.bashrc
```
```
sudo reboot
```

Your VM will be rebooted, wait and reconnect.

Now let's install cudnn 9.2 for CUDA 12.5.

```
chmod +x install-24-04-part2-V2.sh
```
```
sudo ./install-24-04-part2-V2.sh
```
```
source ~/.bashrc
```

Let's install all python pieces of software.

```
chmod +x install-24-04-part3-V2.sh
```
```
sudo ./install-24-04-part3-V2.sh
```
## Testing the framworks

* Set Environment

Run the following commands

```
source /etc/environment
```
```
source ~/.bashrc
```

* Tensorflow, Pytorch

Enter the following command on your terminal

```
python3
```
then in the python command line type 
```
import tensorflow
```

```
import torch
```
```
import torchvision
```
if you did not get any error then exit out from python by exit().



* Pycharm 

Note: Mac users need to acivate [Xquartz](https://www.xquartz.org/) in their machine and then open your terminal. In other words, when you are ssh ing to VM use -X as follows:

```
ssh -X -i <private key file > <user name>@<External Ip address>
``` 

Note: Windows users use Mobaexterma and you are fine.

To activate pycharm enter the following commands 

```
./pycharm.sh
```
* Testing by python file:

Run the following commands

```
source /etc/environment
```
```
source ~/.bashrc
```

Change directory by
```
cd /home/ubuntu/Cloud-Computing/Deep-Learning-Kit-Installation/Shell-Script-Installation/Ubuntu-24.04
```
Run the test.py and check the frameworks.

```
python3 test.py
```
## Getting Started - Option 2

Send me your email address and I add you to my boot disk. You can start your VM and then choose custom image, then pick the class image.
