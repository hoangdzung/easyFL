import torch
from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule

class Model(FModule):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, stride=1, padding=3)
        ) 
        self.flatten = nn.Flatten()
        self.branch1 = nn.Sequential(
            # nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, stride=1, padding=3)
            nn.MaxPool2d(2, 2),
            nn.ReLU(), 
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=3, stride=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
        )
        self.fc1 = nn.Linear(20,10)
            
        self.branch2 = nn.Sequential(
            # nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, stride=1, padding=3)
            nn.MaxPool2d(2, 2),            
            nn.ReLU(),
            nn.Dropout2d(0.5)
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=5, stride=1, padding=3),
            nn.MaxPool2d(2, 2), 
            nn.ReLU(),
            nn.Dropout2d(0.5)
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=3),
            nn.MaxPool2d(2, 2), 
            nn.ReLU(),
            nn.Dropout2d(0.5)
            nn.Flatten(),
            nn.Linear(500,84),
            nn.Dropout1d(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(84,10)
        )
        self.exit_threshold = 0.3
        
    def forward(self, x, n=0):
        x = self.base(x)
        if n ==0:
            x = self.branch1(x)
            x = self.fc1(x)
            return x 
            
        # not_exit = torch.special.entr(F.softmax(x,dim=1)).sum(1) > self.exit_threshold

        # branch_x = x_base[not_exit]
        x = self.branch2(x)
        x = self.fc2(x)
        # x[not_exit, :] =  branch_x

        return x
    
    def pred_and_rep(self, x, n):
        if n ==0:
            x = self.base(x)
            e = self.branch1(x)
            o = self.fc1(e)
            return o, [e]
        else:
            x = self.base(x)
            e = self.branch2(x)
            o = self.fc2(e) 
            return o, [e]

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)