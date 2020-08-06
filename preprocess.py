import cv2
import numpy as np


def preprocess_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 42, 89)
    kernel = np.ones((3, 3))
    dial = cv2.dilate(canny, kernel=kernel, iterations=2)

    return dial
