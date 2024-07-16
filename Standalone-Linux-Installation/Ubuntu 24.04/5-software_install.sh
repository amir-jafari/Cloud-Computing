cd ~
# ----------------- Python 3.10.6  ------------------------------------
sudo apt install -y python3-pip
sudo rm /usr/lib/python3.12/EXTERNALLY-MANAGED
#sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev pkg-config wget
#sudo apt install python3-testresources -y
sudo pip3 install tensorflow[and-cuda]
sudo pip3 install -U scikit-learn
sudo pip3 install torch torchvision torchaudio
sudo pip3 install matplotlib
sudo pip3 install pandas
sudo pip3 install seaborn
sudo pip3 install opencv-python
sudo pip3 install pydotplus
sudo pip3 install gpustat
sudo pip3 install sacred
sudo pip3 install pymongo
sudo pip3 install openpyxl
sudo pip3 install tqdm
sudo pip3 install nltk
sudo pip3 install pyspellchecker
sudo pip3 install -U pip setuptools wheel
sudo pip3 install -U 'spacy[cuda-autodetect]'
python3 -m spacy download en_core_web_sm

sudo pip3 install textacy
sudo pip3 install transformers
sudo pip3 install datasets
sudo pip3 install librosa
sudo pip3 install Jupyter