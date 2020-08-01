import cv2
import numpy as np

# def empty(x):
#     pass
#
# cv2.namedWindow("Parameters")
# cv2.resizeWindow("Parameters",640,240)
# cv2.createTrackbar("Threshold1","Parameters",59,255,empty)
# cv2.createTrackbar("Threshold2","Parameters",41,255,empty)

def preprocess_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    # threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")

    # canny = cv2.Canny(blur,42,89)
    canny = cv2.Canny(blur, 42, 89)
    kernel = np.ones((3, 3))
    dial = cv2.dilate(canny, kernel=kernel, iterations=2)
    # result = cv2.erode(dial, kernel=kernel, iterations=1)
    result = dial
    return result
