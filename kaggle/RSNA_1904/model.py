# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 2019
@author: Subin Joo
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
        
#########  1. VGG-16
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
                
######### 2. custom ResNet-16
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):

    def __init__(self, block, layers, input_channel=1, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc_last = nn.Linear(num_classes+1, 2)
        self.dropOut = nn.Dropout()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, x2): # two input, x2 = patient age
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dropOut(x)
        x = torch.cat((x, x2), 1) # num_classes -> num_classes + 1
        x = self.fc_last(x)
        
        return x
    
def resnet18(**kwargs):
    print("resnet18 model load")
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)   
    
    
    
    
    
    
