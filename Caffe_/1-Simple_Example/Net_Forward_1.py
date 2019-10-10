# %%%%%%%%%%%%% Deep Learning %%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Authors  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Dr. Martin Hagan----->Email: mhagan@okstate.edu 
# Dr. Amir Jafari------>Email: amir.h.jafari@okstate.edu
# %%%%%%%%%%%%% Date:
# V1 Jan - 01 - 2017
# V2 Nov - 12 - 2017
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Caffe  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
############################################CAFFE NET FORWARD EXAMPLE#0###############################################################
import os
import numpy as np
import matplotlib.pyplot as plt
import caffe
import cv2
from PIL import Image


############################################Forward Simulation###############################################################
#my_root = '/home/ajafari/Desktop/Caffe_Files/caffe_example/Net_Forward'
#os.chdir(my_root)
#--------------------------------Set calculation on the Gpu------------------------------------
caffe.set_device(0)
caffe.set_mode_gpu()
#------------------------------Create the Network Architecture---------------------------------
net = caffe.Net('conv.prototxt', caffe.TEST)
#------------------------------Load Data with numpy--------------------------------------------
# Load a gray image of size 1x360x480 (channel x height x width) into the previous net

#im = np.array(Image.open(my_root + '/cat_gray.jpg'))
im = cv2.imread('cat_gray.jpg', 0)
#------------------------------Plot the Input Data--------------------------------------------
plt.figure(1)
plt.imshow(im, cmap="gray")

#------------------------------Some commands to check Data-------------------------------------
type(im)
im.shape


plt.figure(2)
plt.plot(im[1,:])
plt.ylabel('some numbers')
plt.show()

#print (im.shape)
#------------------------------Reshape Data--------------------------------------------------
# Reshape the data blob (1, 1, 100, 100) to the new size (1, 1, 360, 480) to fit the image
#im_input = im[np.newaxis, np.newaxis, :, :]
#net.blobs['data'].reshape(*im_input.shape)
#net.blobs['data'].data[...] = im_input
#------------------------------Forward Simulation------------------------------------------------
# Compute the blobs given this input
#os.system("/home/ajafari/Desktop/draw_net.py conv.prototxt my_net.png")
net.forward()
net.save('mymodel.caffemodel')





