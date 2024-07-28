print(20*'-'+'Checking Deep Learning Frameworks'+20*'-')
import os
import torch
import tensorflow as tf
print(20*'-'+'Nvidia Cuda'+20*'-')
# %%------------------------------------------------------------
os.system("nvidia-smi")
print(20*'.'+20*'.'+20*'.')
os.system("nvcc --version")
# %%------------------------------------------------------------
print(20*'-'+'Torch and Tensorflow Cuda Check'+20*'-')
# %%------------------------------------------------------------
print(20*'.'+20*'.'+20*'.')
if torch.cuda.is_available():
    print(20 * '.' + 20 * '.' + 20 * '.')
    print('Torch Cuda is Available')
    print(20 * '.' + 20 * '.' + 20 * '.')
else:
    print('Torch Cuda is not working!!!!')
    print(20 * '.' + 20 * '.' + 20 * '.')

if  tf.test.gpu_device_name():
    print(20 * '.' + 20 * '.' + 20 * '.')
    print('Tensorflow Cuda is Available')
    print(20 * '.' + 20 * '.' + 20 * '.')
else:
    print(20 * '.' + 20 * '.' + 20 * '.')
    print('Tensorflow Cuda is not working!!!!')
    print(20 * '.' + 20 * '.' + 20 * '.')



print(20*'-'+'END'+20*'-')


