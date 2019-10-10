# %%%%%%%%%%%%% Deep Learning %%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Authors  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Dr. Martin Hagan----->Email: mhagan@okstate.edu 
# Dr. Amir Jafari------>Email: amir.h.jafari@okstate.edu
# %%%%%%%%%%%%% Date:
# V1 Jan - 01 - 2017
# V2 Nov - 12 - 2017
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Caffe  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
import caffe
import types
import lmdb
import numpy as np
import matplotlib.pyplot as plt
############################################Pre Processing Data##############################################################
#my_root = '/home/ajafari/Desktop/Caffe_Files/caffe_example/SquareDiamond'
#os.chdir(my_root)
############################################Pre Processing Data##############################################################

fo = open('diamond.txt', 'r')

file_content = fo.read().strip()
file_content = file_content.replace('\r\n', ';')
file_content = file_content.replace('\n', ';')
file_content = file_content.replace('\r', ';')
diamond = np.matrix(file_content)

fo.close()

fo = open('square.txt', 'r')

file_content = fo.read().strip()
file_content = file_content.replace('\r\n', ';')
file_content = file_content.replace('\n', ';')
file_content = file_content.replace('\r', ';')
square = np.matrix(file_content)

fo.close()

d1 = diamond[np.newaxis, :, :]
d2 = np.roll(diamond,2,axis=0)
d2 = d2[np.newaxis, :, :]
d3 = np.roll(diamond,2,axis=1)
d3 = d3[np.newaxis, :, :]
d4 = np.roll(diamond,-2,axis=0)
d4 = d4[np.newaxis, :, :]
d5 = np.roll(diamond,-2,axis=1)
d5 = d5[np.newaxis, :, :]

s1 = square[np.newaxis, :, :]
s2 = np.roll(square,2,axis=0)
s2 = s2[np.newaxis, :, :]
s3 = np.roll(square,2,axis=0)
s3 = s3[np.newaxis, :, :]
s4 = np.roll(square,2,axis=0)
s4 = s4[np.newaxis, :, :]
s5 = np.roll(square,2,axis=0)
s5 = s5[np.newaxis, :, :]


X = np.array([d1, d2, d3, d4, d5, s1, s2, s3, s4, s5],dtype=np.uint8)
y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0],dtype=np.int64)

N = 10


# We need to prepare the database for the size. We'll set it 10 times
# greater than what we theoretically need. There is little drawback to
# setting this too big. If you still run into problem after raising
# this, you might want to try saving fewer entries in a single
# transaction.
map_size = X.nbytes * 100

env = lmdb.open('testMylmdb', map_size=map_size)

with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(N):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[1]
        datum.height = X.shape[2]
        datum.width = X.shape[3]
        datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
        datum.label = int(y[i])
        str_id = '{:08}'.format(i)

        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())

