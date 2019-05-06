# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 2019
@author: Subin Joo
checkDicom3: add Augmentation, age output
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

# convert grayscale * 3 -> rgb
def convert1Dto3D(imgArr):
    out=[imgArr for i in range(3)]
    return np.stack(out)

# load all files -> seperate abnormal/normal -> shuffle respectively -> merger 6000 abnormal + 6000 normal -> shuffle again
def rearrange(onlyfiles,listId,listLabel):    
    fileName_abnormal,fileName_normal=[],[]
    for oneFile in onlyfiles: # scan dicom file
        if listLabel[listId.index(oneFile[:-4])] == '1': # if dicom file is pneumonia patient
            fileName_abnormal.append(oneFile) # add filename in abnormal list -> total 6012
        else:
            fileName_normal.append(oneFile) # normal list -> total 20672
    
    # abnormal random 6000 datasets + normal random 6000 datasets
    onlyfiles_re=random.sample(fileName_abnormal,6000)+random.sample(fileName_normal,6000)
    
    return random.sample(onlyfiles_re,12000) # random shuffle

class dataLoad01:
    def __init__(self,resize=(224,224),ratioTraining=0.9,numOfDicom=1000,scale="gray"):         
        # step1: load dicom file
        dataDir='stage_2_train_images'
        onlyfiles = [f for f in listdir(dataDir) if isfile(join(dataDir, f))]
        
        # step2: laod csv file
        dataDir_csv='stage_2_train_labels.csv'
        listId,listLabel = [],[]
        with open(dataDir_csv) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, oneline in enumerate(csv_reader):
                if i>0:
                    listId.append(oneline[0]) # Dicom file name
                    listLabel.append(oneline[5]) # patient or not
               
        # step3: shuffle data + balacing data, patients:normal -> 50:50 (nearly, not exact)
        onlyfiles=rearrange(onlyfiles,listId,listLabel)
                
        # step4: create Full image List
        images,labels=[],[]
        ageLabels=[]
        for i, oneFile in enumerate(onlyfiles[:numOfDicom]):
            if oneFile[:-4] in listId:
                if (i%100 == 0): print("in progress : %d / %d " %(i,len(onlyfiles)) )
                
                # load dicom
                filename= dataDir + '/' + oneFile
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
                
                # patient age
                ageLabels.append(int(dataset.PatientAge)/100) # age 0 ~ 100 -> convert 0.0 ~ 1
        
        images = np.stack(images) # 3D -> 4d (1,224,244) -> (:,1,244,244)
        
        size=images.shape[0] # number of total dataset        
        numOfTraining=int(size*ratioTraining)
        
        # separate dataset
        self.images_train=images[:numOfTraining,:,:,:]
        self.images_test=images[numOfTraining:,:,:,:]
                        
        self.labels_train=labels[:numOfTraining]
        self.labels_test=labels[numOfTraining:]
        
        self.age_train=ageLabels[:numOfTraining]
        self.age_test=ageLabels[numOfTraining:]
        
        self.sizeTrain,self.sizeTest = self.images_train.shape[0], self.images_test.shape[0]
        print("train datasets size: "+str(self.images_train.shape))
        print("test datasets size: "+str(self.images_test.shape))
        
        # count number of call
        self.countBatch = 0
        self.countTestData = 0
        
    def batch(self,batch_size=4): # load a few samples of dataset for training
        rangeStart = self.countBatch * batch_size
        rangeEnd = (self.countBatch + 1) * batch_size
        
        selectedImage=self.images_train[rangeStart:rangeEnd,:,:,:]
        labels=np.array(self.labels_train)
        selectedLabel=labels[rangeStart:rangeEnd]
                
        agelabels=np.array(self.age_train)
        selectedAgeLabel=agelabels[rangeStart:rangeEnd]
        
        self.countBatch +=1
        if ( (self.countBatch + 1) * batch_size) > self.sizeTrain: # if rangeEnd is bigger than total size -> initialization
            self.countBatch = 0
            
        return torch.tensor(selectedImage).float(),torch.tensor(selectedLabel).long(),torch.tensor(selectedAgeLabel).float()
        
    def testData(self,batch_size=4):# load a few samples of dataset for testing
        rangeStart = self.countTestData * batch_size
        rangeEnd = (self.countTestData + 1) * batch_size
        
        selectedImage=self.images_test[rangeStart:rangeEnd,:,:,:]
        labels=np.array(self.labels_test)
        selectedLabel=labels[rangeStart:rangeEnd]
                
        agelabels=np.array(self.age_test)
        selectedAgeLabel=agelabels[rangeStart:rangeEnd]
        
        self.countTestData +=1
        if ( (self.countTestData + 1) * batch_size) > self.sizeTest: # if rangeEnd is bigger than total size -> initialization
            self.countTestData = 0
        
        return torch.tensor(selectedImage).float(),torch.tensor(selectedLabel).long().long(),torch.tensor(selectedAgeLabel).float()
    
    def augmentation_Gaussian_noise(self,mu=0.0,sigma=10.0):
        print("\naugmentation : Gaussian noise")
        # train datasets
        addNoise = self.images_train+(np.random.normal(mu, sigma,self.images_train.shape)-128.0)/128.0 # original data + noise
        self.images_train = np.vstack((self.images_train,addNoise))
        
        self.labels_train = self.labels_train + self.labels_train # list + list
        self.age_train = self.age_train + self.age_train # list + list
        
        # test datasets
        addNoise = self.images_test+(np.random.normal(mu, sigma,self.images_test.shape)-128.0)/128.0 # original data + noise
        self.images_test = np.vstack((self.images_test,addNoise))
        
        self.labels_test = self.labels_test + self.labels_test # list + list
        self.age_test = self.age_test + self.age_test # list + list
        
        # update size of datasets
        self.sizeTrain,self.sizeTest = self.images_train.shape[0], self.images_test.shape[0]
        print("train datasets size: "+str(self.images_train.shape))
        print("test datasets size: "+str(self.images_test.shape))

    def augmentation_Image_flip(self):  
        print("\naugmentation : Image flip left<>right")
        # train datasets
        fliped = np.flip(self.images_train, 3) # images_train -> (numOfimage, rgb, y, x) -> flip 4th axis
        self.images_train = np.vstack((self.images_train,fliped))
        
        self.labels_train = self.labels_train + self.labels_train # list + list
        self.age_train = self.age_train + self.age_train # list + list
        
        # test datasets
        fliped = np.flip(self.images_test, 3) # images_test -> (numOftest, rgb, y, x) -> flip 4th axis
        self.images_test = np.vstack((self.images_test,fliped))
        
        self.labels_test = self.labels_test + self.labels_test # list + list
        self.age_test = self.age_test + self.age_test # list + list
        
        # update size of datasets
        self.sizeTrain,self.sizeTest = self.images_train.shape[0], self.images_test.shape[0]
        print("train datasets size: "+str(self.images_train.shape))
        print("test datasets size: "+str(self.images_test.shape))
    
     