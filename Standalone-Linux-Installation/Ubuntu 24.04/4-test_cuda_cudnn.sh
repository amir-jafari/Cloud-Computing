sudo apt install libfreeimage3 libfreeimage-dev -y
sudo cp -r /usr/src/cudnn_samples_v9/ $HOME
cd $HOME/cudnn_samples_v9/mnistCUDNN
sudo make
./mnistCUDNN

