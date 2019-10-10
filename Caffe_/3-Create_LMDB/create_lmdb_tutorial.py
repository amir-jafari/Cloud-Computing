# %%%%%%%%%%%%% Deep Learning %%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Authors  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Dr. Martin Hagan----->Email: mhagan@okstate.edu 
# Dr. Amir Jafari------>Email: amir.h.jafari@okstate.edu
# San Wang ------------>Email: sanwang@gwmail.gwu.edu
# %%%%%%%%%%%%% Date:
# V1 Jan - 01 - 2017
# V2 Nov - 12 - 2017
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Caffe  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
import random
import numpy as np
import fnmatch
import cv2
import caffe
import lmdb

''' Readme First  
The image data set is expected to be located as the following directory structure:
root_dir/train/label_0/image0.jpg
root_dir/train/label_0/image1.jpg
...
root_dir/train/label_1/image0.jpg
root_dir/train/label_1/image1.jpg
...
Same as validation image data set:
root_dir/validation/label_0/image0.jpg
root_dir/validation/label_0/image1.jpg
...
root_dir/validation/label_1/image2.jpg
#################################################################################
Basically you save images from the same category in one folder and name the folder after label
If you have 10 categories, you will have 10 folders with each folder represents one category
'''
#################################################################################
### Things you need to change before using this script on your own image data ###
#################################################################################
# 1.if your image data is not end with .jpg, you need to
# change extension .jpg to the right format appear like the following:
'''
 '*.jpg'
'''
# I only tested on images end with .jpg and .ppm

# 2. if you are using python3, you need to
# change all print function's format

# 3. Put the path that you want to write lmdb files into
# !!!Do not put exist folder here because it will delete these folder every time you run the script
train_lmdb = 'train_lmdb'
validation_lmdb = 'validation_lmdb'

# 4. Put training jpg images path here
JPG_train_path = 'train'
# Put validation jpg images path here
JPG_validation_path = 'validation'

# 4. Put the size you want your images resize to
IMAGE_WIDTH = 80
IMAGE_HEIGHT = 80

# You are ready to run this script now
# ====================================================================================
def transform_img(img, img_width, img_height):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


def make_datum(img, label):
    return caffe.proto.caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, axis=2, start=0).tostring())



# If already exsit previous lmdb folders, remove them
os.system('rm -rf  ' + train_lmdb)
os.system('rm -rf  ' + validation_lmdb)

'''
# for python 3.5+
# there is a quicker way by using glob
# you need to add 'import glob' to do this
train_data = [img for img in glob.glob("root_dir/train/*/*.jpg")]
validation_data = [img for img in glob.glob("root_dir/validation/*/*.jpg")]
'''
# univarsal way
# a list to store all images' path
i = 0
train_data = []
for root, dirnames, filenames in os.walk(JPG_train_path):
    i = i + 1
    for filename in fnmatch.filter(filenames, '*.jpg'):
        train_data.append(os.path.join(root,filename))
num_train = len(train_data)
num_label_train = i - 1

k = 0
validation_data = []
for root, dirnames, filenames in os.walk(JPG_validation_path):
    k = k + 1
    for filename in fnmatch.filter(filenames, '*.jpg'):
        validation_data.append(os.path.join(root,filename))
num_validation = len(validation_data)
num_label_validation = k - 1

#Shuffle train_data
print 'shuffling train data'
random.shuffle(train_data)

print '\nCreating train_lmdb'
env_db = lmdb.open(train_lmdb, map_size=int(1e12))
with env_db.begin(write=True) as txn:
    for idx, img_path in enumerate(train_data):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        label = int(img_path.split('/')[-2])
        datum = make_datum(img, label)
        txn.put('{:0>5d}'.format(idx), datum.SerializeToString())
        print '{:0>5d}'.format(idx) + ':' + img_path
env_db.close()


print '\nCreating validation_lmdb'
env_db = lmdb.open(validation_lmdb, map_size=int(1e12))
with env_db.begin(write=True) as txn:
    for idx, img_path in enumerate(validation_data):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        label = int(img_path.split('/')[-2])
        datum = make_datum(img, label)
        txn.put('{:0>5d}'.format(idx), datum.SerializeToString())
        print '{:0>5d}'.format(idx) + ':' + img_path
env_db.close()

print '\nFinished processing all images'
print '\nTraining data has {} images in {} labels'.format(num_train,
                                                          num_label_train)
print '\nValidation data has {} images in {} labels'.format(num_validation,
                                                            num_label_validation)
