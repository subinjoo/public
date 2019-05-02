# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 2019
@author: Subin Joo
custom utils
"""

# check whether white boxes are inside the red box or not
def ckeckDetectedBoxes(painter,detectedBoxes,redBoxRange,imageReduceRatio):
    for oneBox in detectedBoxes:
        c = [int(i*imageReduceRatio) for i in oneBox] # one detected white box, x min / y min / x max / y max 
        
        if c[0] > redBoxRange[0] and c[1] > redBoxRange[1] and c[2] < redBoxRange[2] and c[3] < redBoxRange[3]:
            painter.drawRect(c[0],c[1],c[2]-c[0],c[3]-c[1])       
        
# check whether the mouse moves left to right or right to left
def checkMouseMovement(x,y,cx,cy):
    returnMovement=[x,y,cx,cy]
    if x > cx:
        returnMovement[0] = cx
        returnMovement[2] = x
    if y > cy:
        returnMovement[1] = cy
        returnMovement[3] = y
    return returnMovement
        
        
    
        
        
    
    