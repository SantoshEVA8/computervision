import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from torch.optim.lr_scheduler import StepLR, OneCycleLR

tfp = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(tfp)
from computervision.deeplearning.models import BasicTransformer
from trainer import train, test
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model =  BasicTransformer().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#scheduler = StepLR(optimizer, step_size=6, gamma=0.1)
scheduler = OneCycleLR(optimizer, max_lr = 0.01, epochs=20, steps_per_epoch=128)

EPOCHS = 20

train_losses = []
test_losses = []
train_acc = []
test_acc = []


for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    tra, trl=train(model, device, trainloader, optimizer, epoch)
    # scheduler.step()
    tsa, tsl=test(model, device, testloader)

    train_losses.append(trl.item())
    test_losses.append(tsl)
    train_acc.append(tra)
    test_acc.append(tsa)


import matplotlib.pyplot as plt

fig, axs = plt.subplots(2,2,figsize=(15,10))
axs[0, 0].plot(train_losses)
axs[0, 0].set_title("Training Loss")
axs[1, 0].plot(train_acc)
axs[1, 0].set_title("Training Accuracy")
axs[0, 1].plot(test_losses)
axs[0, 1].set_title("Test Loss")
axs[1, 1].plot(test_acc)
axs[1, 1].set_title("Test Accuracy")

plt.show()
