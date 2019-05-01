# -*- coding: utf-8 -*-
"""
Created on Wed May  1 2019
@author: Subin Joo
reference https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview
"""

from os import listdir
from os.path import isfile, join

import csv

import pydicom
from pydicom.data import get_testdata_files
import matplotlib.pyplot as plt
import numpy as np

# step1: set dicom file dir
dataDir='.\stage_2_train_images' # dicom folder
onlyfiles = [f for f in listdir(dataDir) if isfile(join(dataDir, f))]

# step2: load csv file
dataDir_csv='stage_2_train_labels.csv' # csv file
listId,listLabel = [],[]
with open(dataDir_csv) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for i, oneline in enumerate(csv_reader):
        if i>0:
            listId.append(oneline[0])
            listLabel.append(oneline[5])  
    
labels=[]
patientAge_normal,patientAge_abnormal=[0]*10,[0]*10
patientSex_normal,patientSex_abnormal=[0,0],[0,0]
for oneFile in onlyfiles:
    filename= dataDir + '\\' + oneFile
    dataset = pydicom.dcmread(filename) # read dicom file
    labels.append(int(listLabel[listId.index(oneFile[:-4])])) # label list
    
    # relate with age, gender
    age = int(int(dataset.PatientAge)/10) # 십의 자리, check 10-digit of age
    gender = dataset.PatientSex # gender of patient
    
    try:
        if int(listLabel[listId.index(oneFile[:-4])]) == 0: # is normal?
            patientAge_normal[age] +=1
            if gender == "M":
                patientSex_normal[0] += 1 # male
            else:
                patientSex_normal[1] += 1 # female
                
        else: # abnormal
            
            patientAge_abnormal[age] +=1
            if gender == "M":
                patientSex_abnormal[0] += 1 # male
            else:
                patientSex_abnormal[1] += 1 # female
    except:
        print(oneFile) # abnormal dicom file, need to check

# plt bar plot
def plot(objects,performance,ylabel,title):
    y_pos = np.arange(len(objects))
    
    plt.bar(y_pos, performance, align='center', alpha=1.0)
    plt.xticks(y_pos, objects)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    
# plot 1, age graph
objects = ['~'+str((i+1)*10) for i in range(10)]
plot(objects,patientAge_normal,ylabel='Number of Subjects',title='Normal')
plot(objects,patientAge_abnormal,ylabel='Number of Patients',title='Patients with lung opacities')

# incidence by age
incidence=np.array(patientAge_abnormal)/(np.array(patientAge_normal)+np.array(patientAge_abnormal))
plot(objects,incidence,ylabel='Attack Rate',title='Patients with lung opacities')

# plot 2, gender graph
objects = ['M','F']
plot(objects,patientSex_normal,ylabel='Number of Subjects',title='gender')
plot(objects,patientSex_abnormal,ylabel='Number of Patients',title='gender')

# incidence by gender
incidence=np.array(patientSex_abnormal)/(np.array(patientSex_normal)+np.array(patientSex_abnormal))
plot(objects,incidence,ylabel='Attack Rate',title='gender')


    




