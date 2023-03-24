import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector
#numpy for matrix multiplication
cap = cv2.VideoCapture(0)
# Number of hands detected
detector = HandDetector(maxHands=1)

offset = 20
imageSize = 300

folder = "imagen/C"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    
    if hands:
        #Working with only one hands detected
        hand =hands[0]
        # w = width of hands , h = height of hands
        x,y,w,h = hand['bbox']
        # In order to have the same image size
        imgWB = np.ones((imageSize, imageSize, 3),np.uint8)*255
        imgCrop =img[y-offset:y + h+offset, x-offset:x + w+offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h/w
        # If height is more than width
        if aspectRatio > 1:
            #K for constant
            k = imageSize / h
            #math ceil for always up
            wCal = math.ceil(k*w)
            # New height
            imgResize = cv2.resize(imgCrop,(wCal,imageSize))
            #with the new height new shape
            imgResizeShape = imgResize.shape
            # Using wGap to put the img in the middle
            wGap = math.ceil((imageSize - wCal)/2)
            # We do not use the channels 
            #We first is 300 so we do not change it
            imgWB[:, wGap:wGap+wCal] = imgResize

         
        
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
        
        #Showing the box
        cv2.imshow("imgCrop", imgCrop)
        cv2.imshow("imgWB", imgWB)
    
    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWB)
        print(counter)