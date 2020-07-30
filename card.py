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

flatten_card_set = []

while True:
    success, img = cap.read()
    imgResult = img.copy()
    imgResult2 = img.copy()

    thresh = preprocess.preprocess_img(img)

    four_corners_set = process.findContours(thresh, imgResult, draw=True)

    flatten_card_set = process.flatten_card(imgResult2, four_corners_set)
    process.get_corner_snip(flatten_card_set)
    cv2.imshow('Result', display.stackImages(0.85, [imgResult, thresh]))

    wait = cv2.waitKey(1)
    if wait & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

# print(flatten_card_set)
# print(four_corners_set[0])
# cv2.imwrite('Warped.png', flatten_card_set[0])
# cv2.imwrite('Warped2.png', flatten_card_set[1])

# cv2.waitKey(0)
