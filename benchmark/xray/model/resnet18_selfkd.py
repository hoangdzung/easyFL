"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""
import torch.nn as nn
from utils.fmodule import FModule

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class Model(FModule):
    def __init__(self, block=BasicBlock, num_block=[2,2,2,2], num_classes=10):
        super().__init__()
        self.in_channels = 64
        self.b012_base_conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.b012_base_conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.b012_base_conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.b12_branch2_conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.b2_branch2_conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.b012_fc0 = nn.Linear(128 * block.expansion, num_classes)
        self.b12_fc1 = nn.Linear(256 * block.expansion, num_classes)
        self.b2_fc2 = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, n=0):
        x = self.b012_base_conv1(x)
        x = self.b012_base_conv2_x(x)
        x = self.b012_base_conv3_x(x)
        if n==0:
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.b012_fc0(x)
            return x 

        x = self.b12_branch2_conv4_x(x)
        if n==1:
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            x = self.b12_fc1(x)
            return x

        x = self.b2_branch2_conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.b2_fc2(x)
        return x

    def pred_and_rep(self, x, n):
        os, es =[], []
        x = self.b012_base_conv1(x)
        x = self.b012_base_conv2_x(x)
        x = self.b012_base_conv3_x(x)
    
        x1 = self.avg_pool(x)
        e1 = x1.view(x1.size(0), -1)
        o1 = self.b012_fc0(e1) 
        os.append(o1)
        es.append(e1)
        
        if n==0:
            return os, es 

        x = self.b12_branch2_conv4_x(x)

        x2 = self.avg_pool(x)
        e2 = x2.view(x2.size(0), -1)
        o2 = self.b12_fc1(e2) 
        os.append(o2)
        es.append(e2)
        
        if n==1:
            return os, es 

        x = self.b2_branch2_conv5_x(x)

        x3 = self.avg_pool(x)
        e3 = x3.view(x3.size(0), -1)
        o3 = self.b2_fc2(e3) 
        os.append(o3)
        es.append(e3)
        return os, es 

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)
