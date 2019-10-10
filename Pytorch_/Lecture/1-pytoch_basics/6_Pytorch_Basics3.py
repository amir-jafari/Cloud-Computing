import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable


# ========================== Using pretrained model ==========================#
# Download and load pretrained resnet.
resnet = torchvision.models.resnet18(pretrained=True)

# If you want to finetune only top layer of the model.
for param in resnet.parameters():
    param.requires_grad = False

# Replace top layer for finetuning.
resnet.fc = nn.Linear(resnet.fc.in_features, 100)

# For test.
images = Variable(torch.randn(10, 3, 224, 224))
outputs = resnet(images)
print(outputs.size())  # (10, 100)

# ============================ Save and load the model ============================#
# Save and load the entire model.
torch.save(resnet, 'model.pkl')
model = torch.load('model.pkl')

# Save and load only the model parameters(recommended).
torch.save(resnet.state_dict(), 'params.pkl')
resnet.load_state_dict(torch.load('params.pkl'))