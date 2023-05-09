import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from cvzone.ClassificationModule import Classifier
from keras.preprocessing.image import ImageDataGenerator
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

target_size = (64, 64)
target_dims = (64, 64, 3)


data_augmentor = ImageDataGenerator(samplewise_center=True, 
                                    samplewise_std_normalization=True)

while True:
    success, img = cap.read()
    hands = detector.findHands(img, draw=False)


    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        cv2.imshow("Image2", imgCrop)

    img_file = skimage.transform.resize(imgCrop, (64, 64, 3))
    img_arr = np.asarray(imgCrop).reshape((-1, 64, 64, 3))

    if img.shape == (64,64,3):
        prediction, index = classifier.getPrediction(img_arr)
        print(prediction, index)

    
    
    cv2.imshow("Image", img)
    cv2.imshow("ImageCropped", img_file)
    cv2.waitKey(1)

