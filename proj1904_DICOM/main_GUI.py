# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 2019
@author: Subin Joo
"""
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np
import sys

import pydicom

from PIL import Image
from PIL.ImageQt import ImageQt

from externalModule.FRCNN import *  # FRCNN model
from utils2 import * # load subfunction

# DICOM image class
class Dicom:
    def __init__(self,fileName):
        dataset = pydicom.dcmread(fileName) # DICOM load           
        DicomData=dataset.pixel_array#, cmap=plt.cm.bone
        
        arr2img=DicomData/16 #3, arraty -> image, 4096 scale -> 256 scale (12 bits -> 8 bits)
        arr2img=arr2img.astype(np.uint8)
        self.img = Image.fromarray(arr2img) # array -> image, DICOM image
        DicomImgOrig=QPixmap.fromImage(ImageQt(self.img)) # image -> pyqt GUI
        DicomImgOrig=DicomImgOrig.scaledToHeight(700) # image resize
        self.DicomImg=DicomImgOrig
        
        self.imageReduceRatio = 700 / self.img.size[1] # reduce ratio 700 / original size
        
        # draw text on the image
        painter = QPainter(DicomImgOrig)
        painter.setPen(QPen(Qt.white))
        painter.setFont(QFont("Arial", 10))
        painter.drawText(QRectF(50.0,500.0,300.0,300.0), Qt.AlignLeft|Qt.AlignTop, self.patientData(dataset)) # patient data
        self.clear = False
        
        # coordinates of detected boxes
        self.detectedBoxes = [[]]
    
        # read patient data from DICOM
    def patientData(self,dataset):
        displayText=(
        "Patient ID: "+dataset.PatientID+
        "\nStudy Date: "+dataset.StudyDate+
        "\nAge: "+str(int(dataset.StudyDate[:4])-int(dataset.PatientBirthDate[:4])))
        return displayText
        
        # draw a red box on the image when click mouse
    def draw(self,x,y,cx,cy):
        self.DicomImg1=self.DicomImg.copy() # clear image by copying origin
        painter = QPainter(self.DicomImg1)
        painter.setPen(QPen(Qt.red, 3, Qt.SolidLine))
        painter.drawRect(x,y,cx-x,cy-y)
        
        # draw white boxes in the red box
    def drawDetectedBoxes(self,redBoxRange):
        painter = QPainter(self.DicomImg1)
        painter.setPen(QPen(Qt.white, 2, Qt.SolidLine))
        
        ckeckDetectedBoxes(painter,self.detectedBoxes,redBoxRange,self.imageReduceRatio) # if the box is located in mouseRedbox, draw
                
# mouse action class
class MouseAction:    
    def mousePressAct(self, QMouseEvent): # mouse click 
        if QMouseEvent.button() == Qt.LeftButton: # left click -> laod DICOM (only first time)
            if self.isFirstOpen:
                self.loadDICOM()
                self.isFirstOpen = False
            else:                               # after second time -> draw a box on the image (True)
                self.drawing = True
                self.x = QMouseEvent.pos().x() # initial position of mouse
                self.y = QMouseEvent.pos().y()        
            
        if QMouseEvent.button() == Qt.RightButton: # right click -> open menu
            self.mousePressActRight(QMouseEvent)           
            
    def mouseMoveAct(self, QMouseEvent): # mouse move
        if Qt.LeftButton and self.drawing and (not self.isFirstOpen):
            self.cx = QMouseEvent.pos().x() # mouse position update
            self.cy = QMouseEvent.pos().y() 
            
            self.DicomImgOOP.draw(self.x,self.y,self.cx,self.cy) # draw a red box              
            self.layout_image.setPixmap(self.DicomImgOOP.DicomImg1) # update GUI  
            
            self.DicomImgOOP.clear = False          
            
    def mouseReleaseAct(self,QMouseEvent): # mouse release
        if QMouseEvent.button() == Qt.LeftButton:
            self.drawing = False            
            isReverse=checkMouseMovement(self.x,self.y,self.cx,self.cy) # check direction of mouse movement
            
            if not self.DicomImgOOP.clear: # clear all boxes?
                self.DicomImgOOP.drawDetectedBoxes(isReverse) # draw detected boxex on the image          
                self.layout_image.setPixmap(self.DicomImgOOP.DicomImg1) # update GUI; plot boxes 

# GUI class
class Main(QWidget,MouseAction,Dicom):
    def __init__(self):
        super().__init__() # bring parent init
        
        # initial parameter
        self.isFirstOpen=True
        self.drawing=False

        layout = QFormLayout() # create sub-main layout
        self.layout_image = QLabel()       
        self.layout_image.setText("\n\n\n\n\n\n\n\nClick hear to load DICOM or Image")
        self.layout_image.setFont(QFont("Times New Roman", 22, QFont.Bold))
        self.layout_image.setAlignment(Qt.AlignCenter)      
        self.layout_image.mousePressEvent = self.mousePressAct       
        self.layout_image.mouseMoveEvent = self.mouseMoveAct               
        self.layout_image.mouseReleaseEvent = self.mouseReleaseAct  
        
        layout.addRow(self.layout_image)
        self.setLayout(layout) 
        self.update()        
        
        # load trained Faster-RCNN model
        self.model=loadTrainedModel() 
        
    def loadDICOM(self): # load DICOM file
        fileName = QFileDialog.getOpenFileName(self, 'OpenFile')[0] # open specific file  
        self.DicomImgOOP=Dicom(fileName) # create DICOM class
        self.layout_image.setPixmap(self.DicomImgOOP.DicomImg) # update GUI        
        
        # teeth detection using faster R-CNN
        self.DicomImgOOP.detectedBoxes = teethObjectDetection(self.DicomImgOOP.img.convert('RGB'),self.model)
        
    def mousePressActRight(self,event): # mouse right click -> open menu
        menu = QMenu(self)
        action_load = menu.addAction("New Dicom")
        action_clear = menu.addAction("Clear image")
        action_bright = menu.addAction("Brightness")
        menu.addSeparator()
        action_func1 = menu.addAction("expansion")
        action_func2 = menu.addAction("clear All")
        action_func3 = menu.addAction("Function 1")
        action_func4 = menu.addAction("Function 2")
        edit = menu.addMenu("Edit     ▶")
        edit.addAction("copy")
        edit.addAction("paste")

        action = menu.exec_(self.mapToGlobal(event.pos()))
        if action == action_load:
            self.loadDICOM() # load new Dicom
        elif action == action_clear:
            self.DicomImgOOP.clear = True 
            self.layout_image.setPixmap(self.DicomImgOOP.DicomImg) # clear image

if __name__ == '__main__':
    app = QApplication([])
    app.setStyle('Fusion')
    app_style = QPalette()
    app_style.setColor(QPalette.Window, QColor(53, 53, 53))
    app_style.setColor(QPalette.WindowText, Qt.white)    
    app.setPalette(app_style)

    window = Main()        
    window.resize(1024, 600)
    window.setWindowTitle("GUI Programing Project version 0.1")
    window.show()
    app.exec_()
    sys.exit()
