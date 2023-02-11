'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os

from computervision.deeplearning.models import *
from utils import progress_bar


class Trainer():
    def __init__(self, lr=0.1, resume=False):

      self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
      self.best_acc = 0  # best test accuracy
      self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch
      self.resume=resume
      # Data
      print('==> Preparing data..')
      self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

      self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

      self.trainset = torchvision.datasets.CIFAR10(
          root='./data', train=True, download=True, transform=self.transform_train)
      self.trainloader = torch.utils.data.DataLoader(
          self.trainset, batch_size=128, shuffle=True, num_workers=2)

      self.testset = torchvision.datasets.CIFAR10(
          root='./data', train=False, download=True, transform=self.transform_test)
      self.testloader = torch.utils.data.DataLoader(
          self.testset, batch_size=100, shuffle=False, num_workers=2)

      self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

      # Model
      print('==> Building model..')

      self.net = ResNet18()
      self.net = self.net.to(self.device)
      if self.device == 'cuda':
          net = torch.nn.DataParallel(self.net)
          cudnn.benchmark = True

      if self.resume:
          # Load checkpoint.
          print('==> Resuming from checkpoint..')
          assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
          checkpoint = torch.load('./checkpoint/ckpt.pth')
          self.net.load_state_dict(checkpoint['net'])
          best_acc = checkpoint['acc']
          start_epoch = checkpoint['epoch']

      self.criterion = nn.CrossEntropyLoss()
      self.optimizer = optim.SGD(net.parameters(), lr=lr,
                          momentum=0.9, weight_decay=5e-4)
      self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)


    # Training
    def train(self,epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


    def test(self,epoch):
        global best_acc
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': self.net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc


    def train_model(self):
        for epoch in range(self.start_epoch, self.start_epoch + 2):
            self.train(epoch)
            self.test(epoch)
            self.scheduler.step()

if __name__ == "__main__":
     model_trainer=Trainer()
     model_trainer.train_model()