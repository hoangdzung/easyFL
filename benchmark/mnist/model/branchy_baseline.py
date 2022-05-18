import torch
from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule

class Model(FModule):
    def __init__(self):
        super().__init__()
        self.base1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, stride=1, padding=3)
        ) 
        self.flatten = nn.Flatten()
        self.branch1 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.ReLU(), 
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, stride=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(20,10)
        )
        
    def forward(self, x):
        x_base = self.base1(x)
        x = self.branch1(x_base)
        return x


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)