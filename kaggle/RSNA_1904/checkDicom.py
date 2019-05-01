# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 2019
@author: Subin Joo

load datasets for train & test
"""
from os import listdir
from os.path import isfile, join

import csv

import pydicom
from pydicom.data import get_testdata_files
from PIL import Image
import numpy as np
import random
import torch

def convert1Dto3D(imgArr):
    out=[imgArr for i in range(3)]
    return np.stack(out)

def rearrange(onlyfiles,listId,listLabel):    
    fileName_abnormal,fileName_normal=[],[]
    for oneFile in onlyfiles: # scan dicom file
        if listLabel[listId.index(oneFile[:-4])] == '1': # if dicom file is pneumonia patient
            fileName_abnormal.append(oneFile) # add filename in abnormal list -> total 6012
        else:
            fileName_normal.append(oneFile) # normal list -> totla 20672
    
    # abnormal random 6000 datasets + normal random 6000 datasets
    onlyfiles_re=random.sample(fileName_abnormal,6000)+random.sample(fileName_normal,6000)
    
    return random.sample(onlyfiles_re,12000) # random suffle


class dataLoad01:
    def __init__(self,resize=(224,224),ratioTraining=0.9,numOfDicom=1000,scale="gray"): 
        # step1: load dicom file
        dataDir='.\stage_2_train_images'
        onlyfiles = [f for f in listdir(dataDir) if isfile(join(dataDir, f))]
        
        # step2: laod csv file
        dataDir_csv='stage_2_train_labels.csv'
        listId,listLabel = [],[]
        with open(dataDir_csv) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, oneline in enumerate(csv_reader):
                if i>0:
                    listId.append(oneline[0])
                    listLabel.append(oneline[5])  
               
        # step3: suffle data, from origin to patients:normal -> 50:50 (nearly, not exact)
        onlyfiles=rearrange(onlyfiles,listId,listLabel)
                
        # step4: create Full image List
        images,labels=[],[]
        for i, oneFile in enumerate(onlyfiles[:numOfDicom]):
            if oneFile[:-4] in listId:
                if (i%100 == 0): print("in progress : %d / %d " %(i,len(onlyfiles)) )
                
                # load dicom
                filename= dataDir + '\\' + oneFile
                dataset = pydicom.dcmread(filename)
                
                # dicom -> image -> resize (with ANTIALIAS) -> array
                arr2img=dataset.pixel_array
                img = Image.fromarray(arr2img)
                img.thumbnail(resize, Image.ANTIALIAS) 
                img=(np.array(img,dtype=np.int16) - 128) / 128 # img > numpy array, normalization
                if scale == "gray":
                    img_reshape=img.reshape((1,img.shape[0],img.shape[1])) # (224,224) -> (1,224,224)
                elif scale == "rgb":
                    img_reshape=convert1Dto3D(img)                         # (224,224) -> (3,224,224)
                
                images.append(img_reshape) # image list
                labels.append(int(listLabel[listId.index(oneFile[:-4])])) # label list
        
        images = np.stack(images) # 3D -> 4d (1,224,244) -> (:,1,244,244)
        
        size=images.shape[0] # number of total dataset        
        numOfTraining=int(size*ratioTraining)
        
        # separate dataset (without random suffle)
        self.images_train=images[:numOfTraining,:,:,:]
        self.images_test=images[numOfTraining:,:,:,:]
                        
        self.labels_train=labels[:numOfTraining]
        self.labels_test=labels[numOfTraining:]
        
        self.sizeTrain,self.sizeTest = numOfTraining, (size-numOfTraining)
        
    def batch(self,batch_size=4): # load a few samples of dataset for training
        selectedNum=random.sample(range(self.images_train.shape[0]), batch_size) # random suffle
        
        selectedImage=self.images_train[selectedNum,:,:,:]
        labels=np.array(self.labels_train)
        selectedLabel=labels[selectedNum]
        return torch.tensor(selectedImage).float(),torch.tensor(selectedLabel).long()
        
    def testData(self,batch_size=4):# load a few samples of dataset for testing
        selectedNum=random.sample(range(self.images_test.shape[0]), batch_size) # random suffle
        
        selectedImage=self.images_test[selectedNum,:,:,:]
        labels=np.array(self.labels_test)
        selectedLabel=labels[selectedNum]
        return torch.tensor(selectedImage).float(),torch.tensor(selectedLabel).long()

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        