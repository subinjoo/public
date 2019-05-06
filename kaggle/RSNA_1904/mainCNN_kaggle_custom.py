# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 2019
@author: Subin Joo
"""
# 1. load test data
import torchvision
from checkDicom3 import *
from utils import *
import matplotlib.pyplot as plt
import numpy as np

# load image and normalization
dataLoad=dataLoad01(resize=(224,224),numOfDicom=4000,scale="rgb") # load dataset (4000,3,224,224)
#dataLoad=dataLoad01(resize=(224,224),numOfDicom=4000,scale="gray") # load dataset (4000,1,224,224)

# data augmentation
#dataLoad.augmentation_Gaussian_noise()
#dataLoad.augmentation_Image_flip()

classes = ('Normal', 'Patient')

## get some random training images
images, labels, _ = dataLoad.batch()

# show images & print labels
imshow(torchvision.utils.make_grid(images)) # imshow -> utils
print(' '.join('%s' % classes[labels[j]] for j in range(4)))

# 2. Define a Convolutional Neural Network
import torch.nn as nn

## custom model
#from model import * # load custom model
#net = resnet18(input_channel=3) # load custom model
#modelpath="resnet18-5c106cde.pth" # download from 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
#myModel_kvpair = pretraining(net,modelpath) # pretraining -> utils
#net.load_state_dict(myModel_kvpair)

# torchvision model
from torchvision import models # laod pretrained ResNet
net = models.resnet18(pretrained=True)

# gpu or cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# 3. Define a Loss function and optimizer
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adagrad(net.parameters(), lr=0.001)

import time
startTime=time.time() # training start time

# 4. Train the network
print("\n\n start training \n\n")
batch_size = 4
train_loss,train_acc=[],[]
val_loss,val_acc=[],[]
for epoch in range(40):  # loop over the dataset multiple times
    print("epoch: ",epoch)

    running_loss = 0.0
    correct,total = 0,0
    for i in range(int(dataLoad.sizeTrain/batch_size)):
        # get the input datas      
        inputs, labels, ageLabels = dataLoad.batch(batch_size=batch_size) # load random training datasets
        inputs, labels  = inputs.to(device), labels.to(device)
        
        ageLabels=torch.tensor([[i] for i in ageLabels]) # patient age data
        ageLabels=ageLabels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
#        outputs = net(inputs,ageLabels) # train custom model (two inputs)
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()  

        # print statistics
        running_loss += loss.item()
        if i % 300 == 299:    # print every 300 mini-batches
            print('[%d, %d] loss: %.3f' %(epoch + 1, i + 1, running_loss / i))
                
    train_loss.append(running_loss / i)
    train_acc.append(correct / total)
    
    ## 5. Test the network on the test data
    #  the network performs on the whole validation dataset
    correct,total = 0,0
    batch_size = 4
    testing_loss = 0.0
    with torch.no_grad():
        for i in range(int(dataLoad.sizeTest/batch_size)):
            # get the input datas      
            images, labels, ageLabels = dataLoad.testData(batch_size=batch_size)
            images, labels = images.to(device), labels.to(device)
            
            ageLabels=torch.tensor([[i] for i in ageLabels]) # patient age data
            ageLabels=ageLabels.to(device)
            
#            outputs = net(images,ageLabels) # train custom model (two inputs)
            outputs = net(images)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()           
            
            loss_test = criterion(outputs, labels)
            testing_loss += loss_test.item()
            
    print(testing_loss / i)
    val_loss.append(testing_loss / i)
    val_acc.append(correct / total)

print('\n\n Finish Training')
print('training time: ', time.time()-startTime,'\n\n')

# plot loss and acuracy
plt.plot(train_loss)
plt.plot(val_loss)
plt.show()

plt.plot(train_acc)
plt.plot(val_acc)
plt.show()

# accuracy in indicidual classes
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))
batch_size = 4
with torch.no_grad():
    for i in range(int(dataLoad.sizeTest/batch_size)):
        images, labels, ageLabels = dataLoad.testData(batch_size=batch_size)
        images, labels = images.to(device), labels.to(device)
        
        ageLabels=torch.tensor([[i] for i in ageLabels])
        ageLabels=ageLabels.to(device)
        
#        outputs = net(images,ageLabels)
        outputs = net(images)
        
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(2):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
    
    
