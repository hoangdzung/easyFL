from numpy import expm1
import torch
from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule

class Model(FModule):
    def __init__(self):
        super().__init__()
        self.b01_layer0 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(2),
        )

        self.b01_layer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.b01_layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.b01_layer21 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=8),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
        )

        self.b1_layer21 = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=8),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
        )
        self.relu = nn.ReLU()
        self.b01_gap = torch.nn.AdaptiveAvgPool2d(1)
        self.b01_flatten = nn.Flatten()
        self.b01_fc = torch.nn.Linear(128, 10)

    def forward(self, x, n=0):
        x = self.b01_layer0(x)
        x = self.b01_layer1(x)
        x = self.b01_layer2(x)
        if n ==0:
            x = x+ self.b01_layer21(x) *2
        else:
            x = x +  self.b01_layer21(x) + self.b1_layer21(x)
        x = self.relu(x)    
        x = self.b01_gap(x)
        x = self.b01_flatten(x)
        x = self.b01_fc(x)
        return x


    def pred_and_rep(self, x, n):
        x = self.b01_layer0(x)
        x = self.b01_layer1(x)
        x = self.b01_layer2(x)
        if n ==0:
            x = x+ self.b01_layer21(x) *2
        else:
            x = x +  self.b01_layer21(x) + self.b1_layer21(x)
            
        x = self.b01_gap(x)
        e = self.b01_flatten(x)
        o = self.b01_fc(e)
        return [o], [e]


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)