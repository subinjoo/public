# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 2019

@author: Subin Joo

Simple CNN test for kaggle competition
reference https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview
"""
# 1. load test data
import torch
import torchvision
import torchvision.transforms as transforms
from checkDicom import *

# load image and normalization -> [-1,1]
dataLoad=dataLoad01(resize=(224,224),numOfDicom=10,scale="rgb") # laod dataset

classes = ('Normal', 'Patient')

# show some of the training images, for fun.
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

## get some random training images
images, labels = dataLoad.batch()

# show images & print labels
imshow(torchvision.utils.make_grid(images))
print(' '.join('%s' % classes[labels[j]] for j in range(4)))

# 2. Define a Convolutional Neural Network
import torch.nn as nn
import torch.nn.functional as F
from model import * # load custom model

from torchvision import models # laod pretrained model VGG-16 model
from torchsummary import summary

#net = models.resnet18(pretrained=False) # pytorch model

net = vgg16(input_channels=1) # custom model
summary(net, (1, 224, 224)) # summary model

# gpu or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# 3. Define a Loss function and optimizer
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

import time
startTime=time.time()

# 4. Train the network
print("\n\n start training \n\n")
for epoch in range(10):  # loop over the dataset multiple times
    print("epoch: ",epoch)

    running_loss = 0.0
    for i in range(dataLoad.sizeTrain):
        # get the inputs
        
        inputs, labels = dataLoad.batch(batch_size=4) # load random training datasets
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

print('\n\n Finished Training')
print('training time: ', time.time()-startTime,'\n\n')

## 5. Test the network on the test data
images, labels = dataLoad.testData(batch_size=4) # load random test datasets

# show images & print labels
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%s' % classes[labels[j]] for j in range(4)))

outputs = net(images.to(device)) # test image into net model
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%s' % classes[predicted[j]] for j in range(4)))

#  the network performs on the whole dataset.
correct = 0
total = 0
with torch.no_grad():
    for i in range(dataLoad.sizeTest):
        images, labels = dataLoad.testData(batch_size=4)
        images, labels = images.to(device), labels.to(device)
        
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of test images: %d' % ( 100 * correct / total))
   


