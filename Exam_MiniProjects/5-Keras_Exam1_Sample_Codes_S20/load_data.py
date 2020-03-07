import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder

if "train" not in os.listdir():
    os.system("wget https://storage.googleapis.com/exam-deep-learning/train.zip")
    os.system("unzip train.zip")

DATA_DIR = os.getcwd() + "/train/"
RESIZE_TO = 50

x, y = [], []
for path in [f for f in os.listdir(DATA_DIR) if f[-4:] == ".png"]:
    x.append(cv2.resize(cv2.imread(DATA_DIR + path), (RESIZE_TO, RESIZE_TO)))
    with open(DATA_DIR + path[:-4] + ".txt", "r") as s:
        label = s.read()
    y.append(label)
x, y = np.array(x), np.array(y)
le = LabelEncoder()
le.fit(["red blood cell", "ring", "schizont", "trophozoite"])
y = le.transform(y)
print(x.shape, y.shape)
