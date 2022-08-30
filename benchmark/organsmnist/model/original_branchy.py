import torch
from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule

class Model(FModule):
    def __init__(self):
        super().__init__()
        self.base = nn.Conv2d(1, 5, kernel_size=5, stride=1, padding=3)

        self.branch1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(5, 10, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.branch1_fc = nn.Linear(490, 10)

        self.branch2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(5, 10, kernel_size=5, stride=1, padding=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.branch2_fc = nn.Linear(500, 10)


    def forward(self, x, n=1):
        x = self.base(x)
        if n==0:
            x = self.branch1(x)
            x = self.branch1_fc(x)
        else:
            x = self.branch2(x)
            x = self.branch2_fc(x)
        return x

    def pred_and_rep(self, x, n):
        x = self.base(x)
        if n==0:
            e = self.branch1(x)
            o = self.branch1_fc(e)
        else:
            e = self.branch2(x)
            o = self.branch2_fc(e)
        return o, [e]

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)