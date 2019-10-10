import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets


# ======================== Loading data from numpy ========================#
a = np.array([[1, 2], [3, 4]])
b = torch.from_numpy(a)
c = b.numpy()

# ===================== Implementing the input pipline =====================#
# Download and construct dataset.
train_dataset = dsets.CIFAR10(root='../data/', train=True, transform=transforms.ToTensor(), download=True)

# Select one data pair (read data from disk).
image, label = train_dataset[0]
print(image.size())
print(label)

# Data Loader (this provides queue and thread in a very simple way).
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True,num_workers=2)

# When iteration starts, queue and thread start to load dataset from files.
data_iter = iter(train_loader)

# Mini-batch images and labels.


images, labels = data_iter.next()

# Actual usage of data loader is as below.
for images, labels in train_loader:
    # Your training code will be written here
    pass


# ===================== Input pipline for custom dataset =====================#
# You should build custom dataset as below.
class CustomDataset(data.Dataset):
    def __init__(self):
        # TODO
        # 1. Initialize file path or list of file names.
        pass

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0

        # Then, you can just use prebuilt torch's data loader.


custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=100,
                                           shuffle=True,
                                           num_workers=2)