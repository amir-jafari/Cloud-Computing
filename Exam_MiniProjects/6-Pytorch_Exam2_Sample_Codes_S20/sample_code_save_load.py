import torch
import torch.nn as nn
from torchvision import datasets

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (3, 3))
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(16, 32, (3, 3))
        self.convnorm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d((2, 2))
        self.linear1 = nn.Linear(32*5*5, 400)
        self.linear1_bn = nn.BatchNorm1d(400)
        self.drop = nn.Dropout(0.5)
        self.linear2 = nn.Linear(400, 10)
        self.act = torch.relu

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
        return self.linear2(x)

data_train = datasets.MNIST(root='.', train=True, download=True)
x_train, y_train = data_train.data.view(len(data_train), 1, 28, 28).float().to(device), data_train.targets.to(device)
data_test = datasets.MNIST(root='.', train=False, download=True)
x_test, y_test = data_test.data.view(len(data_test), 1, 28, 28).float().to(device), data_test.targets.to(device)

model = CNN().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=5e-2)
criterion = nn.CrossEntropyLoss()

for epoch in range(1):
    model.train()
    for batch in range(len(x_train)//512 + 1):
        inds = slice(batch*512, (batch+1)*512)
        optimizer.zero_grad()
        logits = model(x_train[inds])
        loss = criterion(logits, y_train[inds])
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), "cnn.pt")

model.load_state_dict(torch.load("cnn.pt"))
model.eval()
print(model)
