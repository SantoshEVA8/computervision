import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import numpy as np

dropout_value=0.1

class Ultimus(nn.Module):

    def __init__(self):
        super(Ultimus, self).__init__()
        self.linear_k = nn.Linear(48, 8)
        self.linear_q = nn.Linear(48, 8)
        self.linear_v = nn.Linear(48, 8)
        self.linear_out = nn.Linear(8, 48)

    def AM(self,q,k):
        am=torch.matmul(q.transpose(-2, -1), k)
        am=F.softmax(am, dim=1)
        am=am/ np.power(8, 0.5)
        return am

    def Z(self, v, am):
        z=torch.matmul(v, am)
        return z

    def forward(self, x):
        k = self.linear_k(x)
        q = self.linear_q(x)
        v = self.linear_v(x)

        am = self.AM(q, k)
        z = self.Z(v, am)

        out = self.linear_out(z)
        return out

class BasicTransformer(nn.Module):

    def __init__(self):
        super(BasicTransformer, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.Dropout(dropout_value)
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=32)
        )

        self.ultimusblock=nn.Sequential(
            Ultimus(),
            Ultimus(),
            Ultimus(),
            Ultimus()
        )

        self.fc_out=nn.Linear(48,10)
        self.linear_k = nn.Linear(48, 8)


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.gap(x)
        x= x.view(-1, 48)
        x=self.ultimusblock(x)
        x=self.fc_out(x)
        return F.log_softmax(x, dim=-1)

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = BasicTransformer().to(device)
    summary(model, input_size=(3, 32, 32))