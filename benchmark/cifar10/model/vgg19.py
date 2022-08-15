from torch import nn
class Model(nn.Module):
    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        
        # Block 1
        self.b012_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.b12_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.b012_maxpool_1 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # Block 2 
        self.b012_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.b12_layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.b012_maxpool_2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # Block 3 
        self.b012_layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.b012_layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.b12_layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.b2_layer8 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.b012_maxpool_3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        # BLock 4
        self.b012_layer9 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.b012_layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.b12_layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.b2_layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.b012_maxpool_4 = nn.MaxPool2d(kernel_size = 2, stride = 2)
         
        # BLock 5
        self.b012_layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.b012_layer14 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.b12_layer15 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.b2_layer16 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.b012_maxpool_5 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
       
        self.b012_fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU())
        self.b012_fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU())
        self.b012_fc2= nn.Sequential(
            nn.Linear(512, num_classes))
        
    def forward(self, x,n=2):
        out = self.b012_layer1(x)
        out = self.b12_layer2(out)
        out = self.b012_maxpool_1(out)

        out = self.b012_layer3(out)
        out = self.b12_layer4(out)
        out = self.b012_maxpool_2(out)

        out = self.b012_layer5(out)
        out = self.b012_layer6(out)
        out = self.b12_layer7(out)
        out = self.b2_layer8(out)
        out = self.b012_maxpool_3(out)

        out = self.b012_layer9(out)
        out = self.b012_layer10(out)
        out = self.b12_layer11(out)
        out = self.b2_layer12(out)
        out = self.b012_maxpool_4(out)

        out = self.b012_layer13(out)
        out = self.b012_layer14(out)
        out = self.b12_layer15(out)
        out = self.b2_layer16(out)
        out = self.b012_maxpool_5(out)
        
        out = out.reshape(out.size(0), -1)
        out = self.b012_fc(out)
        out = self.b012_fc1(out)
        out = self.b012_fc2(out)
        return out
