import torch
from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule

class Model(FModule):
    def __init__(self):
        super().__init__()
        self.base_layer0 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5, stride=1, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(5),
            nn.ReLU()
        )

        self.base_layer1 = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.base_layer2= nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.base_gap = torch.nn.AdaptiveAvgPool2d(1)
        self.base_flatten = nn.Flatten()
        self.base_fc = torch.nn.Linear(20, 10)

        self.branch2_layer3 = nn.Sequential(
            nn.Conv2d(20, 40, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(40),
            nn.ReLU()
        )
        self.branch2_fc = torch.nn.Linear(40, 10)

    def forward(self, x, n=0):
        x = self.base_layer0(x)
        x = self.base_layer1(x)
        x = self.base_layer2(x)
    
        if n==0:
            x = self.base_gap(x)
            x = self.base_flatten(x)
            x = self.base_fc(x) 
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
            x = self.base_gap(x)
            e = self.base_flatten(x)
            o = self.base_fc(e) 
            return [o], [e]
        else:
            x1 = self.base_gap(x)
            e1 = self.base_flatten(x1)
            o1 = self.base_fc(e1) 
            x2 = self.branch2_layer3(x)
            x2 = self.base_gap(x2)
            e2 = self.base_flatten(x2)
            o2 = self.branch2_fc(e2)
        return [o1, o2], [e1, e2]
        

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)