import torch
from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = input + shortcut
        return nn.ReLU()(input)

class Model(FModule):
    def __init__(self):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5, stride=1, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(5),
            nn.ReLU()
        )

        self.layer11 = nn.Sequential(
            nn.Conv2d(5, 10, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.layer12 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
        )

        self.layer21 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.layer22 = nn.Sequential(
            nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = torch.nn.Linear(20, 10)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer11(x)
        x = x + self.layer12(x)
        x = self.layer21(x)
        x = x+ self.layer22(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)