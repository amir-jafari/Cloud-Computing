print(20*'-'+'Checking Deep Learning Frameworks'+20*'-')
import os
print(20*'-'+'Nvidia Cuda'+20*'-')
# %%------------------------------------------------------------
os.system("nvidia-smi")
print(20*'.'+20*'.'+20*'.')
os.system("nvcc --version")
print(20*'.'+20*'.'+20*'.')
os.chdir('/home/ubuntu/cudnn_samples_v7/mnistCUDNN')
os.system("./mnistCUDNN")
# %%------------------------------------------------------------
print(20*'-'+'Python Packages'+20*'-')
# %%------------------------------------------------------------
try:
    import matplotlib
    print('matplotlib' + '---->' + 'Test Passed')
except:
    print('matplotlib IS NOT INSTALLED')
print(20*'.'+20*'.'+20*'.')
# %%------------------------------------------------------------
try:
    import numpy
    print('numpy' + '---->' + 'Test Passed')
except:
    print('numpy IS NOT INSTALLED')
print(20*'.'+20*'.'+20*'.')
# %%------------------------------------------------------------
try:
    import torch
    print('torch' + '---->' + 'Test Passed')
except:
    print('torch IS NOT INSTALLED')
print(20*'.'+20*'.'+20*'.')
try:
    import tensorflow
    print('tensorflow' + '---->' + 'Test Passed')
except:
    print('tensorflow IS NOT INSTALLED')
print(20*'.'+20*'.'+20*'.')
try:
    import keras
    print('keras' + '---->' + 'Test Passed')
except:
    print('keras IS NOT INSTALLED')
print(20*'.'+20*'.'+20*'.')
try:
    import pandas
    print('pandas' + '---->' + 'Test Passed')
except:
    print('pandas IS NOT INSTALLED')
print(20*'.'+20*'.'+20*'.')
try:
    import scipy
    print('scipy' + '---->' + 'Test Passed')
except:
    print('scipy IS NOT INSTALLED')
print(20*'.'+20*'.'+20*'.')

try:
    import cv2
    print('cv2' + '---->' + 'Test Passed')
except:
    print('cv2 IS NOT INSTALLED')
print(20*'.'+20*'.'+20*'.')
try:
    import lmdb
    print('lmdb' + '---->' + 'Test Passed')
except:
    print('lmdb IS NOT INSTALLED')
print(20*'.'+20*'.'+20*'.')

try:
    import sympy
    print('sympy' + '---->' + 'Test Passed')
except:
    print('sympy IS NOT INSTALLED')

print(20*'-'+'END'+20*'-')


