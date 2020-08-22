# Debian 9 Deep Learning Software shell script

## Option2

*  Get a GCP (Get a GPU optimized pre installed cuda 10 and cudnn 7.6)

*  To install all the Frameworks (tensorflow, theano, pytorch, Keras), launch your VM  and run the following commands in order in to your VM terminal (CUDA 10.0 and cudnn 7.6).

```
sudo apt install git -y
```
```
git clone https://github.com/amir-jafari/Cloud-Computing.git
```
```
cd Cloud-Computing/Deep-Learning-Kit-Installation/Shell-Script-Installation/Debian-9/
```
We are going to install Version 3:

```
mv install-Debian-9-GCP-V1.sh ~
```
```
cd ~
```
```
chmod +x install-Debian-9-GCP-V1.sh
```
```
sudo ./install-Debian-9-GCP-V1.sh
```

## Testing the framworks

* Set Environment

Run the following commands


* Tensorflow, Keras, Theano (Virtual Python 3)

Enter the following command on your terminal

```
python3
```
then in the python env write
```
import tensorflow
```
```
import keras
```
```
import theano
```
if you did not get any error then exit out from python by exit().



* Pytorch 

Enter the following command on your terminal
```
python3
```
then in the python env write
```
import torch
```
```
import torchvision
```

if you did not get any error then exit out from python env by exit(). 


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


Change directory by
```
cd Cloud-Computing/Deep-Learning-Kit-Installation/Shell-Script-Installation/Debian-9/
```
Run the test.py and check the frameworks.

```
python3 test_GCP.py
```
