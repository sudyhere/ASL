import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from cvzone.ClassificationModule import Classifier

import skimage 

cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands = 1)
classifier = Classifier("Model/HSRmodel.h5","Model/labels.txt")
offset = 20
imgSize = 300
counter = 0
labels = ["A", "B", "C", "D","del", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N","nothing", "O", "P", "Q", "R", "S","space", "T", "U", "V", "W", "X", "Y", "Z"
]
imageSize = 64

while True:
    success, img = cap.read()
    hands = detector.findHands(img, draw=False)

    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize,imgSize, 3), np.uint8)*255
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]


        aspectRatio = h/w

        if aspectRatio>1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((300 - wCal)/2)
            imgWhite[:,wGap:wCal+wGap] = imgResize
            img_file = skimage.transform.resize(img, (imageSize, imageSize, 3))
            img_arr = np.asarray(img_file).reshape((-1, imageSize, imageSize, 3))
            prediction, index = classifier.getPrediction(img_arr)
            print(prediction, index)


        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop,(imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((300 - hCal)/2)
            imgWhite[hGap:hCal+hGap, :] = imgResize



        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)

