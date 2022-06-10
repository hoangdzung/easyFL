import torch
from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule

class Model(FModule):
    def __init__(self):
        super().__init__()
        self.b012_layer0 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.b012_layer1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.b12_layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            # nn.Dropout2d(0.4),
            nn.ReLU(),
            nn.MaxPool2d(2),

        )

        self.b2_layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            # nn.Dropout2d(0.7),
            nn.ReLU(),
            nn.MaxPool2d(2),

        )
        
        self.b012_gap = torch.nn.AdaptiveAvgPool2d(1)
        self.b012_flatten = nn.Flatten()
        self.b012_fc = torch.nn.Linear(64, 10)
        self.b12_fc = torch.nn.Linear(128, 10)
        self.b2_fc = torch.nn.Linear(256, 10)

    def forward(self, x, n=0):
        x = self.b012_layer0(x)
        x = self.b012_layer1(x)
    
        if n==0:
            x = self.b012_gap(x)
            x = self.b012_flatten(x)
            x = self.b012_fc(x) 
            return x

        x = self.b12_layer2(x)
        if n==1:
            x = self.b012_gap(x)
            x = self.b012_flatten(x)
            x = self.b12_fc(x) 
            return x

        x = self.b2_layer3(x)
        x = self.b012_gap(x)
        x = self.b012_flatten(x)
        x = self.b2_fc(x) 
        return x
             

    def pred_and_rep(self, x, n):
        os, es =[], []
        x = self.b012_layer0(x)
        x = self.b012_layer1(x)
    
        x1 = self.b012_gap(x)
        e1 = self.b012_flatten(x1)
        o1 = self.b012_fc(e1) 
        os.append(o1)
        es.append(e1)
        
        if n==0:
            return os, es 

        x = self.b12_layer2(x)

        x2 = self.b012_gap(x)
        e2 = self.b012_flatten(x2)
        o2 = self.b12_fc(e2) 
        os.append(o2)
        es.append(e2)
        
        if n==1:
            return os, es 

        x = self.b2_layer3(x)

        x3 = self.b012_gap(x)
        e3 = self.b012_flatten(x3)
        o3 = self.b2_fc(e3) 
        os.append(o3)
        es.append(e3)
        return os, es 

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)