import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class MyModel_test(nn.Module):
    def __init__(self):
        super(MyModel_test, self).__init__()
        #self.cnn_block1 = CNNBlock(2, 32)  # 从2个通道开始
        #self.cnn_block2 = CNNBlock(32, 64)
        #self.cnn_block3 = CNNBlock(64, 128)
        self.flatten = nn.Flatten()
        #self.fc1 = nn.Linear(1152, 512)  # Adjust size according to output of conv layers
        #self.fc1 = nn.Linear(576,256)
        self.fc1 = nn.Linear(450,128)
        self.fc2 = nn.Linear(128, 5)  # Assume some number of output classes
        

    def forward(self, x):
        # x = self.cnn_block1(x)
        # x = F.max_pool2d(x, 2)  # Downsample
        # x = self.cnn_block2(x)
        # x = F.max_pool2d(x, 2)  # Downsample
        # #x = self.cnn_block3(x)
        # x = self.flatten(x)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)


        return x

