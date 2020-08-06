import cv2
import time
from opencv_card_recognizer import preprocess
from opencv_card_recognizer import display
from opencv_card_recognizer import process
from opencv_card_recognizer import model_wrapper
from opencv_card_recognizer import constants

frameWidth = 640
frameHeight = 480

# change to 1 if using USB webcam
cap = cv2.VideoCapture(0)
frame_rate = 30

# width is id number 3, height is id 4
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# change brightness to 150
cap.set(10, 150)

flatten_card_set = []

# get the model corresponding to ranks and suits
modelRanks, modelSuits = model_wrapper.model_wrapper('imgs/ranks', constants.NUM_RANKS, 'rankWeights.h5'), \
                         model_wrapper.model_wrapper('imgs/suits', constants.NUM_SUITS, 'suitWeights.h5'),

prev = 0

while True:
    time_elapsed = time.time() - prev

    success, img = cap.read()

    if time_elapsed > 1. / frame_rate:
        prev = time.time()

        imgResult = img.copy()
        imgResult2 = img.copy()

        # preprocess the image
        thresh = preprocess.preprocess_img(img)
        # find the set of corners that represent the cards
        four_corners_set = process.findContours(thresh, imgResult, draw=True)
        # warp the corners to form an image of the cards
        flatten_card_set = process.flatten_card(imgResult2, four_corners_set)
        # get a crop of the borders for each of the cards
        cropped_images = process.get_corner_snip(flatten_card_set)
        # isolate the rank and suits from the cards
        rank_suit_mapping = process.split_rank_and_suit(cropped_images)
        # figure out what the suits and ranks might be for the cards
        pred = process.eval_rank_suite(rank_suit_mapping, modelRanks, modelSuits)
        # display the text results on the card
        process.show_text(pred, four_corners_set, imgResult)

        # show the overall image
        cv2.imshow('Result', display.stackImages(0.85, [imgResult, thresh]))

    wait = cv2.waitKey(1)
    if wait & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
