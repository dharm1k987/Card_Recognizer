import cv2
import numpy as np

def findContours(img, original, draw=False):
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    four_corners_set = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, closed=True)

        if area > 2000:
            approx = cv2.approxPolyDP(cnt, 0.02*perimeter, closed=True)

            numCorners = len(approx)

            if draw and numCorners == 4:
                # cv2.drawContours(original, cnt, -1, (255, 0, 0), 3)

                # create bounding box around shape
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)

                four_corners_set.append(approx)
                for a in approx:
                    cv2.circle(original, (a[0][0], a[0][1]), 10, (255, 0, 0), 3)
                    print((a[0][0], a[0][1]))
                print("___________")

    return four_corners_set

def flatten_card(img, set_of_corners):
    width, height= 200, 300
    img_outputs = []

    for corner_set in set_of_corners:
        # define 4 corners of the King of Spades card
        pts1 = np.float32(corner_set)
        # now define which corner we are referring to
        pts2 = np.float32([[0,0],[0,height],[width,height],[width,0]])

        # transformation matrix
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        imgOutput = cv2.warpPerspective(img, matrix, (width, height))
        img_outputs.append(imgOutput)

    return img_outputs

def get_corner_snip(flattened_images):
    c = 0
    for img in flattened_images:
        # img have shape (300, 200)
        # crop the image in half first, and then the width in half again
        crop = img[10:100, 0:30]

        # threshold the corner
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 80, 80)
        kernel = np.ones((3, 3))
        dial = cv2.dilate(canny, kernel=kernel, iterations=2)
        result = cv2.erode(dial, kernel=kernel, iterations=1)

        cv2.imwrite('Image' + str(c) + '.png', result)
        c += 1

