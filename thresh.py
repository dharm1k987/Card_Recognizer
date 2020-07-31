import cv2
from opencv_card_recognizer import display
import numpy as np

frameWidth = 640
frameHeight = 480

cap = cv2.VideoCapture(0)

# width is id number 3, height is id 4
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# change brightness to 150
cap.set(10, 150)

def empty(x):
    pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("Threshold1","Parameters",55,255,empty)
cv2.createTrackbar("Threshold2","Parameters",73,255,empty)
cv2.createTrackbar("Area","Parameters",5000,30000,empty)

while True:
    success, img = cap.read()

    imgResult = img.copy()

    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")

    imgCanny = cv2.Canny(imgGray,threshold1,threshold2)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)



    cv2.imshow("Result", display.stackImages(0.8, [img, imgDil]))


    wait = cv2.waitKey(1)
    if wait & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()