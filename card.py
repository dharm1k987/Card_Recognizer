import cv2
from opencv_card_recognizer import preprocess
from opencv_card_recognizer import display
from opencv_card_recognizer import process
from opencv_card_recognizer import augtest

frameWidth = 640
frameHeight = 480

cap = cv2.VideoCapture(0)

# width is id number 3, height is id 4
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# change brightness to 150
cap.set(10, 150)

flatten_card_set = []

modelRanks, modelSuits = augtest.model_wrapper('imgs/ranks', 13, 'ranksOvernightWeights2.h5'),\
                         augtest.model_wrapper('imgs/suits', 4, 'suitsOvernightWeights2.h5'),

while True:
    success, img = cap.read()
    imgResult = img.copy()
    imgResult2 = img.copy()

    thresh = preprocess.preprocess_img(img)

    four_corners_set = process.findContours(thresh, imgResult, draw=True)

    flatten_card_set = process.flatten_card(imgResult2, four_corners_set)
    cropped_images = process.get_corner_snip(flatten_card_set)
    rank_suit_mapping = process.split_rank_and_suit(cropped_images)
    pred = process.eval_rank_suite(rank_suit_mapping, modelRanks, modelSuits)
    process.show_text(pred, four_corners_set, imgResult)
    cv2.imshow('Result', display.stackImages(0.85, [imgResult, thresh]))

    wait = cv2.waitKey(1)
    if wait & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()