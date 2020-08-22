# Ubuntu 14.04 or 16.04 steps to run the shell script

## Getting Started 

Version 1: Installs all Python 2.7 and 3.5, Pycharm, R and Rstudio with packages needed, launch your VM  and run the following commands in order in to your VM terminal.

```
sudo apt-get install git -y
```
```
git clone https://github.com/amir-jafari/Cloud-Computing.git
```
```
cd Cloud-Computing/ML-Installation/

```
We are going to insatll  Version 1 installation:
```
mv install-ML1-final_V1.sh ~
```
```
cd ~
```
```
chmod +x install-ML1-final_V1.sh
```

If you are ussing AWS use just ubuntu as net id if you are using GCP use your GWU net id.

```
sudo ./install-ML1-final_V1.sh
```

## Testing the framworks

* Python

Run the following commands

```
source /etc/environment
```
```
source ~/.bashrc
```
then to test Python

```
python
```

exit out from python env by exit()

* R

Type the following command in your terminal.

```
R
```

* Pycharm and Rstudio

Note: Mac users need to acivate [Xquartz](https://www.xquartz.org/) in their machine and then open your terminal. In other words, when you are ssh ing to VM use -X as follows:

```
ssh -X -i <private key file > <netid>@<External Ip address>
``` 

Note: Windows users use Mobaexterma and you are fine.

To activate pycharm enter the following commands:

```
pycharm-community
```

To activate pycharm enter the following commands:

```
rstudio
```




