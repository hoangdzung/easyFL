import torch
from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule

class Model(FModule):
    def __init__(self):
        super().__init__()
        self.base_layer0 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=3, stride=1, padding=3),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(2),

        )

        self.base_layer1 = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=3, stride=1, padding=3),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.base_layer2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=3),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.branch2_layer3 = nn.Sequential(
            nn.Conv2d(20, 40, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(40),
            nn.ReLU(),
        )
        self.base_gap = torch.nn.AdaptiveAvgPool2d(1)
        self.base_flatten = nn.Flatten()
        self.branch2_fc = torch.nn.Linear(40, 10)
        self.branch1_fc = torch.nn.Linear(20, 10)

    def forward(self, x, n=0):
        x = self.base_layer0(x)
        x = self.base_layer1(x)
        x = self.base_layer2(x)
    
        if n==0:
            x = self.base_gap(x)
            x = self.base_flatten(x)
            x = self.branch1_fc(x) 
        else:
            x = self.branch2_layer3(x)
            x = self.base_gap(x)
            x = self.base_flatten(x)
            x = self.branch2_fc(x)
        return x

    def pred_and_rep(self, x, n):
        x = self.base_layer0(x)
        x = self.base_layer1(x)
        x = self.base_layer2(x)
    
        if n==0:
            e = self.base_flatten(x)
            x = self.base_gap(x)
            x = self.base_flatten(x)
            o = self.branch1_fc(x) 
            return o, [e]
        else:
            e1 = self.base_flatten(x)
            x = self.branch2_layer3(x)
            e2 = self.base_flatten(x)
            x = self.base_gap(x)
            x = self.base_flatten(x)
            o = self.branch2_fc(x)
        return o, [e1, e2]
        

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)