import torch
from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule

class Model(FModule):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, stride=1, padding=3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(), 
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        ) 
        self.flatten = nn.Flatten()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=1, kernel_size=3, stride=1, padding=0),
            nn.Flatten(),
            nn.Linear(36,10)
        )
                
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=5, stride=1, padding=3),
            nn.MaxPool2d(2, 2), 
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=3),
            nn.MaxPool2d(2, 2), 
            nn.Flatten(),
            nn.Linear(180,10),
        )
        self.exit_threshold = 0.3
        
    def forward(self, x, n=0):
        x_base = self.base(x)
        if n ==0:
            x = self.branch1(x_base)
            return x 
            
        # not_exit = torch.special.entr(F.softmax(x,dim=1)).sum(1) > self.exit_threshold

        # branch_x = x_base[not_exit]
        x = self.branch2(x_base)
        # x[not_exit, :] =  branch_x

        return x
    
    def pred_and_rep(self, x, n):
        if n ==0:
            x = self.base1(x)
            e = self.flatten(x)
            o = self.branch1(x)
            return o, [e]
        else:
            x = self.base1(x)
            e = self.flatten(x)
            o = self.branch2(x)     
            return o, [e]

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)