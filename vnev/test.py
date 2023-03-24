import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
#numpy for matrix multiplication
cap = cv2.VideoCapture(0)
# Number of hands detected
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")

offset = 20
imageSize = 300

folder = "imagen/C"
counter = 0

#Letras /Class
# Also can import labels.txt 
Labels = ["A","B","C"]

while True:
    success, img = cap.read()
    #image out put 
    imgOutPut =img.copy()
     # We use false in order to hide the red points 
     #but we don't want to do it that way
     #because is going to afect the classifier
    hands, img = detector.findHands(img)
    
    if hands:
        #Working with only one hands detected
        hand =hands[0]
        # w = width of hands , h = height of hands
        x,y,w,h = hand['bbox']
        # In order to have the same image size
        imgWB = np.ones((imageSize, imageSize, 3),np.uint8)*255
        #to capture well the imagen
        imgCrop =img[y-offset:y + h+offset, x-offset:x + w+offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h/w
        # If height is more than width
        if aspectRatio > 1:
            #K for constant
            k = imageSize / h
            #math ceil for always up
            wCal = math.ceil(k*w)
            # New width
            imgResize = cv2.resize(imgCrop,(wCal,imageSize))
            #with the new width new shape /unnecesary?
            imgResizeShape = imgResize.shape
            # Using wGap to put the img in the middle
            wGap = math.ceil((imageSize - wCal)/2)
            # We do not use the channels 
            #We first is 300 so we do not change it
            imgWB[:, wGap:wGap+wCal] = imgResize
            #Classifier draw the predicion in green for default
            #we use false in other to void that
            prediction, index = classifier.getPrediction(imgWB,draw=False)
            #print(predition,index)
            #Using fot testing in terminal
            
            

         

        else:
            #K for constant
            k = imageSize / w
            #math ceil for always up
            hCal = math.ceil(k*h)
            # New height
            imgResize = cv2.resize(imgCrop,(imageSize,hCal))
            #with the new height new shape / this funcion is unnecessary
            imgResizeShape = imgResize.shape
            # Using wGap to put the img in the middle
            hGap = math.ceil((imageSize - hCal)/2)
            # We do not use the channels and use the new resize
            imgWB[ hGap:hGap+hCal, :] = imgResize
            prediction, index = classifier.getPrediction(imgWB,draw=False)
            
        
        #Showing the box
        cv2.imshow("imgCrop", imgCrop)
        cv2.imshow("imgWB", imgWB)
        #Full color /BackGround 
        cv2.rectangle(imgOutPut,(x-offset,y-offset-50),(x-offset+90,y-offset),(255,0,255),cv2.FILLED)
        #Letters, size  , color and thickness
        cv2.putText(imgOutPut,Labels[index],(x,y-26),cv2.FONT_HERSHEY_COMPLEX,1.7,(255,255,255),2)
        cv2.rectangle(imgOutPut,(x-offset,y-offset),(x+offset+w,y+offset+h),(255,0,255),4)
    
    cv2.imshow('Image', imgOutPut)
    key = cv2.waitKey(1)
   