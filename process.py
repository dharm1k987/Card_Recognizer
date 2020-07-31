import cv2
import numpy as np

def findContours(img, original, draw=False):
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    four_corners_set = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, closed=True)

        if area > 2000:
            approx = cv2.approxPolyDP(cnt, 0.01*perimeter, closed=True)

            numCorners = len(approx)

            if draw and numCorners == 4:
                # cv2.drawContours(original, cnt, -1, (255, 0, 0), 3)

                # create bounding box around shape
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # make sure the image is oriented right top left, bot left, bot right, top right
                l1 = np.array(approx[0]).tolist()
                l2 = np.array(approx[1]).tolist()
                l3 = np.array(approx[2]).tolist()
                l4 = np.array(approx[3]).tolist()

                # print([l1,l2,l3,l4])

                # find which ones have similar x values
                similarX = []
                sortedX = sorted([l1,l2,l3,l4], key=lambda x: x[0][0])
                # sortedX[0] and sortedX[1] are the left half
                # the one with the smaller y value goes first

                finalOrder = []
                finalOrder.extend(sorted(sortedX[0:2], key=lambda x: x[0][1]))

                # now sortedX[1] and sortedX[2] are the right half
                # the one with the larger y value goes first
                finalOrder.extend(sorted(sortedX[2:4], key=lambda x: x[0][1], reverse=True))
                print(approx)
                print(finalOrder)

                four_corners_set.append(finalOrder)
                for a in approx:
                    cv2.circle(original, (a[0][0], a[0][1]), 10, (255, 0, 0), 3)
                    # print((a[0][0], a[0][1]))
                # print(approx)
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
        kernel = np.ones((4, 4))
        result = cv2.dilate(canny, kernel=kernel, iterations=3)
        # result = cv2.erode(dial, kernel=kernel, iterations=1)


        cv2.imwrite('Image' + str(c) + '-before-warp.png', img)
        cv2.imwrite('Image' + str(c) + '.png', result)
        c += 1

        cropped_images.append(result)

    return cropped_images

def split_rank_and_suit(cropped_images):
    imgC = 0
    rank_suit_mapping = []
    for img in cropped_images:
        # print(img.shape)

        # find the two largest contours
        contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        highest_two = dict()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                cv2.fillPoly(img, pts=[cnt], color=0)
                continue
            perimeter = cv2.arcLength(cnt, closed=True)
            highest_two[area] = [cnt, perimeter]

        # select the largest two in this image
        cntC = 0
        mapping = []
        # if len(list(highest_two.keys())) != 2:
        #     return []
        # print(highest_two.keys())
        for area in sorted(highest_two)[0:2]:
            cnt = highest_two[area][0]
            perimeter = highest_two[area][1]
            approx = cv2.approxPolyDP(cnt, 0.02*perimeter, closed=True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(img, (0, y), (img.shape[0], y+h), (255, 0, 0), 2)
            crop = img[y:y+h][:]
            # cv2.imwrite('{}-{}-crop.png'.format(imgC, cntC), crop)
            cv2.imwrite('{}-{}-reg.png'.format(imgC, cntC), img)

            cntC += 1
            mapping.append([crop, y])

        imgC += 1

        # # # lets store rank and then suit
        # mapping.sort(key=lambda x: x[1])
        # # print(mapping)
        # for m in mapping:
        #     del m[1]
        # # print(mapping)
        # print("-------------")
        #
        # # [ mapping.pop(1) for m in mapping ]
        # # # now we don't need the last item so we can remove
        # if mapping:
        #     rank_suit_mapping.append([mapping[0][0], mapping[1][0]])
        #     cv2.imwrite('{}-0-split.png'.format(imgC), mapping[0][0])
        #     cv2.imwrite('{}-1-split.png'.format(imgC), mapping[1][0])


    return rank_suit_mapping

def eval_rank_suite(rank_suit_mapping):
    # 70x125
    # print(len(rank_suit_mapping))
    c = 0
    return
    for rank, suit in rank_suit_mapping:
        # rank = cv2.resize(rank, (70, 125))
        # suit = cv2.resize(suit, (70, 125))
        cv2.imwrite('{}-0.png'.format(c), rank)
        cv2.imwrite('{}-1.png'.format(c), suit)
        c += 1
