import numpy as np
import torch
import torchvision
import tensorflow as tf
import sklearn
import matplotlib
import pandas as pd
import cv2
import tqdm
import transformers

print("=" * 50)
print("Ubuntu 26.04 Deep Learning Environment Test")
print("=" * 50)

print(f"NumPy:        {np.__version__}")
print(f"PyTorch:      {torch.__version__}")
print(f"TorchVision:  {torchvision.__version__}")
print(f"TensorFlow:   {tf.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"Matplotlib:   {matplotlib.__version__}")
print(f"Pandas:       {pd.__version__}")
print(f"OpenCV:       {cv2.__version__}")
print(f"Transformers: {transformers.__version__}")

print("\n--- GPU Status ---")
print(f"CUDA available (PyTorch): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU (PyTorch):            {torch.cuda.get_device_name(0)}")
print(f"GPUs (TensorFlow):        {tf.config.list_physical_devices('GPU')}")