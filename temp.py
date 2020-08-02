import cv2
import numpy as np
from opencv_card_recognizer import augtest
from opencv_card_recognizer import model

model.loadData('imgs/temp')



# modelRanks, modelSuits = augtest.model_wrapper('imgs/ranks2', 13, 'rank2Weights.h5'), augtest.model_wrapper('imgs/suits2', 4, 'suits2Weights.h5'),
#
#
# img = cv2.imread('2-0-split.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
# print(img.shape)
#
# # img = img[:,0:75]
# # cv2.imshow('img', img)
#
# img = cv2.resize(img, (100, 120))
#
# # cv2.imshow('img', img)
#
# img2 = cv2.imread('2-1-split.png')
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY);
# img2 = cv2.resize(img2, (100, 120))
#
# rankNames = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
# suitNames = ['Hearts', 'Spades', 'Clubs', 'Diamonds']
#
# print(rankNames[np.argmax(augtest.model_predict(modelRanks, img, 'ranks'), axis=0)])
# print(suitNames[np.argmax(augtest.model_predict(modelSuits, img2, 'suits'), axis=0)])

cv2.waitKey(0)