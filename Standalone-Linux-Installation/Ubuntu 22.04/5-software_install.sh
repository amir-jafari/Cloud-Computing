cd ~
# ----------------- Python 3.10.6  ------------------------------------
sudo apt install -y python3-pip
sudo apt install build-essential libssl-dev libffi-dev python3-dev -y
sudo apt-get install tcl-dev tk-dev python-tk python3-tk -y
sudo pip3 install --upgrade pip

sudo apt install python3-testresources -y
sudo -H pip3 install tensorflow
sudo -H pip3 install -U scikit-learn
sudo pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

sudo -H pip3 install matplotlib
sudo -H pip3 install pandas
sudo -H pip3 install seaborn

sudo -H pip3 install leveldb

sudo -H pip3 install opencv-python
sudo -H pip3 install pydotplus
sudo -H pip3 install gpustat
sudo -H pip3 install sacred
sudo -H pip3 install pymongo
sudo -H pip3 install openpyxl
sudo -H pip3 install tqdm


sudo -H pip3 install nltk
sudo -H pip3 install pyspellchecker

pip3 install -U pip setuptools wheel
pip3 install -U 'spacy[cuda-autodetect]'
python3 -m spacy download en_core_web_sm

sudo -H pip3 install textacy
sudo -H pip3 install transformers
sudo -H pip3 install datasets