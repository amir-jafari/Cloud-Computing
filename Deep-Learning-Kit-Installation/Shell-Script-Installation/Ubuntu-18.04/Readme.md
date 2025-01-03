# Ubuntu 18.04 Deep Learning Software Shell Script GUIDE

## Getting Started - Option 1
Latest Version: To install all the Frameworks (tensorflow, theano, pytorch, Keras), launch your VM  and run the following commands in order in to your VM terminal 
Cuda 10.1 Cudnn 7.6

```
sudo apt install git -y
```
```
git clone https://github.com/amir-jafari/Cloud-Computing.git
```
```
cd Cloud-Computing/Deep-Learning-Kit-Installation/Shell-Script-Installation/Ubuntu-18.04/
```
We are going to install Version 3:

```
mv install-18-04-part1-V3.sh ~
```
```
mv install-18-04-part2-V3.sh ~
```
```
cd ~
```
```
chmod +x install-18-04-part1-V3.sh
```
```
sudo ./install-18-04-part1-V3.sh
```
By this time you should have CUDA 10.1 installed correctly. Your VM will be rebooted, wait and reconnect.

Now lets install cudnn 7.6 for CUDA 10.1 and all the softwares.

```
chmod +x install-18-04-part2-V3.sh
```
```
sudo ./install-18-04-part2-V3.sh
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

* Tensorflow, Keras, Theano, Pytorch

Enter the following command on your terminal

```
python3
```
then in the python command line type 
```
import tensorflow
```
```
import keras
```
```
import theano
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
ssh -X -i <private key file > <netid>@<External Ip address>
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
cd /home/ubuntu/Cloud-Computing/Deep-Learning-Kit-Installation/Shell-Script-Installation/Ubuntu-18.04
```
Run the test.py and check the frameworks.

```
python3 test.py
```
## Getting Started - Option 2

Send me your email address and I add you to my boot disk. You can start your VM and then choose custom image, then pick the class image.
