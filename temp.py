import cv2
import numpy as np
from opencv_card_recognizer import augtest

modelRanks, modelSuits = augtest.model_wrapper('imgs/ranks', 13, 'rankWeights.h5'), augtest.model_wrapper('imgs/suits', 4, 'suitWeights.h5'),


img = cv2.imread('imgs/ranks/3-4.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
print(img.shape)

# img = img[:,0:75]
# cv2.imshow('img', img)

img = cv2.resize(img, (100, 120))

cv2.imshow('img', img)

img2 = cv2.imread('0-1.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY);
img2 = cv2.resize(img2, (100, 120))

print(augtest.model_predict(modelRanks, img, 'ranks'))
print(augtest.model_predict(modelSuits, img2, 'suits'))

cv2.waitKey(0)