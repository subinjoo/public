# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 2019

@author: Subin Joo
1. simple CNN: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#define-a-convolutional-neural-network
"""

import torch.nn as nn
import torch.nn.functional as F

# 1. simple CNN model
class CNN(nn.Module):
    def __init__(self,input_channels=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
# 2. VGG-16 model
class vgg16(nn.Module):
    def __init__(self,input_channels=3,init_weights=True):
        super(vgg16, self).__init__()
        
        # convolutional part
        self.convLayers = nn.Sequential(
        nn.Conv2d(input_channels,64,3,padding=1),nn.ReLU(True),
        nn.Conv2d(64,64,3,padding=1),nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        nn.Conv2d(64,128,3,padding=1),nn.ReLU(True),
        nn.Conv2d(128,128,3,padding=1),nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        nn.Conv2d(128,256,3,padding=1),nn.ReLU(True),
        nn.Conv2d(256,256,3,padding=1),nn.ReLU(True),
        nn.Conv2d(256,256,3,padding=1),nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        nn.Conv2d(256,512,3,padding=1),nn.ReLU(True),
        nn.Conv2d(512,512,3,padding=1),nn.ReLU(True),
        nn.Conv2d(512,512,3,padding=1),nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        nn.Conv2d(512,512,3,padding=1),nn.ReLU(True),
        nn.Conv2d(512,512,3,padding=1),nn.ReLU(True),
        nn.Conv2d(512,512,3,padding=1),nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        
        nn.AdaptiveAvgPool2d((7, 7))) # convert image size, n x m -> 7 x 7 using average pooling
        
        # fully connected part
        self.fcs = nn.Sequential(
        nn.Linear(7 * 7 * 512, 4096),nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 1000)
        )      
        
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        x=self.convLayers(x)        
        x = x.view(x.size(0), -1)
        x = self.fcs(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

#modelTest=vgg16()
#model_url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
#modelTest.load_state_dict(model_zoo.load_url(model_url))
    
    
    
    
    
    
    
    
    
