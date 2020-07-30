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
    cropped_images = []
    for img in flattened_images:
        # img have shape (300, 200)
        # crop the image in half first, and then the width in half again
        crop = img[10:100, 0:30]

        # resize by a factor of 4
        crop = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        # threshold the corner
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        canny = cv2.Canny(blur, 50, 50)
        kernel = np.ones((3, 3))
        dial = cv2.dilate(canny, kernel=kernel, iterations=3)
        result = cv2.erode(dial, kernel=kernel, iterations=1)



        cv2.imwrite('Image' + str(c) + '.png', result)
        c += 1

        cropped_images.append(result)

    return cropped_images

def split_rank_and_suit(cropped_images):
    imgC = 0
    for img in cropped_images:
        print(img.shape)

        # find the two largest contours
        contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        highest_two = dict()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, closed=True)
            highest_two[area] = [cnt, perimeter]

        # select the largest two in this image
        cntC = 0
        for area in sorted(highest_two)[0:2]:
            cnt = highest_two[area][0]
            perimeter = highest_two[area][1]
            approx = cv2.approxPolyDP(cnt, 0.02*perimeter, closed=True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(img, (0, y), (img.shape[0], y+h), (255, 0, 0), 2)
            cv2.imwrite('{}-{}.png'.format(imgC, cntC), img)
            cntC += 1
        imgC += 1




    #         approx = cv2.approxPolyDP(cnt, 0.02*perimeter, closed=True)
    #
    #         numCorners = len(approx)
    #
    #         if draw and numCorners == 4:
    #             # cv2.drawContours(original, cnt, -1, (255, 0, 0), 3)
    #
    #             # create bounding box around shape
    #             x, y, w, h = cv2.boundingRect(approx)
    #             cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #


    # contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # four_corners_set = []
    #
    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     perimeter = cv2.arcLength(cnt, closed=True)
    #
    #     if area > 2000:
    #         approx = cv2.approxPolyDP(cnt, 0.02*perimeter, closed=True)
    #
    #         numCorners = len(approx)
    #
    #         if draw and numCorners == 4:
    #             # cv2.drawContours(original, cnt, -1, (255, 0, 0), 3)
    #
    #             # create bounding box around shape
    #             x, y, w, h = cv2.boundingRect(approx)
    #             cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #
    #             four_corners_set.append(approx)
    #             for a in approx:
    #                 cv2.circle(original, (a[0][0], a[0][1]), 10, (255, 0, 0), 3)
    #                 print((a[0][0], a[0][1]))
    #             print("___________")
