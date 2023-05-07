import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from cvzone.ClassificationModule import Classifier


cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands = 1)
classifier = Classifier("Model/HSRmodel.h5","Model/labels.txt")
offset = 20
imgSize = 200
counter = 0
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']


while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize,imgSize, 3), np.uint8)*255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]


        aspectRatio = h/w

        if aspectRatio>1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(200,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((300 - wCal)/2)
            imgWhite[:,wGap:wCal+wGap] = imgResize


        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop,(imgSize, 200))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((300 - hCal)/2)
            imgWhite[hGap:hCal+hGap, :] = imgResize



        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)

