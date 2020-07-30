import cv2
import numpy as np

img = cv2.imread('Image0.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);


contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

highest_two = dict()
for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, closed=True)
    approx = cv2.approxPolyDP(cnt, 0.02*perimeter, closed=True)
    x, y, w, h = cv2.boundingRect(approx)
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)

cv2.imshow('img', img)


cv2.waitKey(0)
