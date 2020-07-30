import cv2
import numpy as np

from opencv_card_recognizer import preprocess
from opencv_card_recognizer import display
from opencv_card_recognizer import process

frameWidth = 640
frameHeight = 480

cap = cv2.VideoCapture(0)

# width is id number 3, height is id 4
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# change brightness to 150
cap.set(10, 150)

while True:
    success, img = cap.read()
    imgResult = img.copy()

    thresh = preprocess.preprocess_img(img)

    process.findContours(thresh, imgResult, draw=True)

    cv2.imshow('Result', display.stackImages(0.85, [imgResult, thresh]))

    wait = cv2.waitKey(1)
    if wait & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
