# -*- coding: utf-8 -*-
"""
Created on Sun May 5 2019
@author: Subin Joo
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# load pretrained model's weights
# !wget -c https://download.pytorch.org/models/resnet18-5c106cde.pth
def pretraining(net,modelpath):
    preTrainedModel=torch.load(modelpath)
    
    trainedModelItems=list(preTrainedModel.items())
    
    tm_layer_name = [i for i,j in trainedModelItems] # get layers name from .pth file
    tm_weights=[j for i,j in trainedModelItems] # get layers weight from .pth file
    
    myModel_kvpair=net.state_dict() # layer names & weights of my model
    for key,value in myModel_kvpair.items(): # step1, check one layer's name and one layer's weight of mine
        try:
            index=tm_layer_name.index(key) # step2, insert one layer's name -> downloaded model
            
            if index == 0: # step 3, get weights from downloaded model -> insert my model
                
                # bring one kernel's weight among three depth of kernel,
                # because my resnet model takes gray scale image (depth = 1)
                if net.conv1.weight.size()[1] == 3:
                    myModel_kvpair[key]=tm_weights[index] # imput image = rgb
                else:
                    myModel_kvpair[key]=tm_weights[index][:,:1,:,:]  # imput image = gray
            else:
                myModel_kvpair[key]=tm_weights[index] 
        except:
            pass
        
    return myModel_kvpair