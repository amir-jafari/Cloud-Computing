# Ubuntu 16.04 virtual env steps to run the shell script

## Getting Started 

Version 1: To install all the Frameworks (torch, caffe, tensorflow, theano, pytorch, Keras), launch your VM  and run the following commands in order in to your VM terminal (CUDA 8).

Version 2: To install all the Frameworks (torch, caffe, tensorflow, theano, pytorch, Keras), launch your VM  and run the following commands in order in to your VM terminal (CUDA 9).


```
sudo apt-get install git -y
```
```
git clone https://github.com/amir-jafari/Cloud-Computing.git
```
```
cd Cloud-Computing/Deep-Learning-Kit-Installation/Shell-Script-Installation/Ubuntu-16.04/
```
We are going to install Version 2:

```
mv install-16-04-final-V2.sh ~
```
```
cd ~
```
```
chmod +x install-16-04-final-V2.sh
```
```
sudo ./install-16-04-final-V2.sh <netid or username of ssh key>
```

## Testing the framworks

* Torch

Run the following commands

```
source /etc/environment
```
```
source ~/.bashrc
```
then to test torch just enter
```
sudo ~/torch/install/bin/luarocks install torch 
```

```
th
```
in to your terminal and you should be able to see the torch environment. Now enter 
```
require 'dp'
```
If torch is is installed correctly then you should be see a long list.

If the GPU is intsalled correctly then after entering the following command you sould see the true output.

```
require 'cunn'

```

```
exit
```
and then enter y to get out of the torch env.

* Caffe (Python 2.7)

Enter the following command on your terminal

```
python
```
then in the python env write
```
import caffe
```
exit out from python env by exit()

* Tensorflow, Keras, Theano (Virtual Python 2.7)

Enter the following command on your terminal
```
source ~/python2/bin/activate
```
```
python
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
if you did not get any error then exit out from python env by exit(). Enter the following command to get out of the virtualenv.
```
deactivate
```


* Pytorch (python 2.7)

Enter the following command on your terminal
```
python
```
then in the python env write
```
import torch
```
```
import torchvision
```

if you did not get any error then exit out from python env by exit(). 


* Tensorflow, Keras, Theano (Virtual python 3.5)

Enter the following command on your terminal
```
source ~/python3/bin/activate
```
```
sudo pip3 install pandas --upgrade
```
```
sudo pip3 install --upgrade numexpr
```
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
if you did not get any error then exit out from python env by exit().  Enter the following command to get out of the virtualenv.
```
deactivate
```


* Pytorch (python 3.5)

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

if you did not get any error then exit out pycharm-communityfrom python env by exit().

* Pycharm and ZeroBraneStudio

Note: Mac users need to acivate [Xquartz](https://www.xquartz.org/) in their machine and then open your terminal. In other words, when you are ssh ing to VM use -X as follows:

```
ssh -X -i <private key file > <netid>@<External Ip address>
``` 

Note: Windows users use Mobaexterma and you are fine.

To activate pycharm enter the following commands 

```
pycharm.sh
```

To activate ZeroBraneStudio

```
zbstudio
```
