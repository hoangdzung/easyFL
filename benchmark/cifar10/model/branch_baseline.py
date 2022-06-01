import torch
from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule

class Model(FModule):
    def __init__(self):
        super().__init__()
        self.base_layer0 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding='valid'),
            nn.MaxPool2d(kernel_size=2, stride=1, padding='valid'),
            nn.ReLU()
        )

        self.base_layer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='valid'),
            nn.MaxPool2d(kernel_size=2, stride=1, padding='valid'),
            nn.ReLU()
        )

        self.base_gap = torch.nn.AdaptiveAvgPool2d(1)
        self.base_flatten = nn.Flatten()
        self.base_fc = torch.nn.Linear(64, 10)

    def forward(self, x, n=0):
        x = self.base_layer0(x)
        x = self.base_layer1(x)
        x = self.base_gap(x)
        x = self.base_flatten(x)
        x = self.base_fc(x)
        
        return x
        
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)