import cv2
import numpy as np
from opencv_card_recognizer import augment
from opencv_card_recognizer import augtest
from opencv_card_recognizer import constants


def findContours(img, original, draw=False):
    # find the set of contours on the threshed image
    contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # sort them by highest area
    proper = sorted(contours, key=cv2.contourArea, reverse=True)

    four_corners_set = []

    for cnt in proper:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, closed=True)

        # only select those contours with a good area
        if area > 2000:
            # find out the number of corners
            approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, closed=True)
            numCorners = len(approx)

            if numCorners == 4:
                # create bounding box around shape
                x, y, w, h = cv2.boundingRect(approx)

                if draw:
                    cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # make sure the image is oriented right: top left, bot left, bot right, top right
                l1 = np.array(approx[0]).tolist()
                l2 = np.array(approx[1]).tolist()
                l3 = np.array(approx[2]).tolist()
                l4 = np.array(approx[3]).tolist()

                finalOrder = []

                # sort by X vlaue
                sortedX = sorted([l1, l2, l3, l4], key=lambda x: x[0][0])

                # sortedX[0] and sortedX[1] are the left half
                finalOrder.extend(sorted(sortedX[0:2], key=lambda x: x[0][1]))

                # now sortedX[1] and sortedX[2] are the right half
                # the one with the larger y value goes first
                finalOrder.extend(sorted(sortedX[2:4], key=lambda x: x[0][1], reverse=True))

                four_corners_set.append(finalOrder)

                if draw:
                    for a in approx:
                        cv2.circle(original, (a[0][0], a[0][1]), 10, (255, 0, 0), 3)

    return four_corners_set


def flatten_card(img, set_of_corners):
    width, height = 200, 300
    img_outputs = []

    for corner_set in set_of_corners:
        # get the 4 corners of the card
        pts1 = np.float32(corner_set)
        # now define which corner we are referring to
        pts2 = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

        # transformation matrix
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        imgOutput = cv2.warpPerspective(img, matrix, (width, height))
        img_outputs.append(imgOutput)

    return img_outputs

def get_corner_snip(flattened_images):
    corner_images = []

    for img in flattened_images:

        # crop the image to where the corner might be
        crop = img[10:constants.CARD_HEIGHT, 0:35]

        # resize by a factor of 4
        crop = cv2.resize(crop, None, fx=4, fy=4)

        # threshold the corner
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
        canny = cv2.Canny(bilateral, 40, 24)
        kernel = np.ones((3, 3))
        result = cv2.dilate(canny, kernel=kernel, iterations=2)

        # append the thresholded image and the original one
        corner_images.append([result, gray])

    return corner_images


def split_rank_and_suit(cropped_images):
    # CAN MAKE THIS FUNCTION SIMPLER

    imgC = 0
    rank_suit_mapping = []

    for img, original in cropped_images:

        # find the contours (we want the rank and suit contours)
        contours, hier = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # find the largest two contours
        highest_two = dict()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # if area < 2000, its not of relevance to us, so just fill it with black
            if area < 2000:
                cv2.fillPoly(img, pts=[cnt], color=0)
                continue
            perimeter = cv2.arcLength(cnt, closed=True)
            # append the contour and the perimeter
            highest_two[area] = [cnt, perimeter]

        # select the largest two in this image
        cntC = 0
        mapping = []


        for area in sorted(highest_two)[0:2]:
            cnt = highest_two[area][0]
            perimeter = highest_two[area][1]
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, closed=True)
            x, y, w, h = cv2.boundingRect(approx)
            crop = original[y:y + h][:]

            sharpened = augment.contrast(crop, 30)

            for i in range(sharpened.shape[0]):
                for j in range(sharpened.shape[1]):
                    if sharpened[i, j] < 150:
                        sharpened[i, j] = max(0, sharpened[i, j] - 60)
                    if sharpened[i, j] > 150:
                        sharpened[i, j] = 255

            cntC += 1
            mapping.append([sharpened, y])

        imgC += 1

        # store rank and then suit
        mapping.sort(key=lambda x: x[1])

        for m in mapping:
            del m[1]

        # # now we don't need the last item so we can remove
        if mapping and len(mapping) == 2:
            rank_suit_mapping.append([mapping[0][0], mapping[1][0]])

    return rank_suit_mapping


def eval_rank_suite(rank_suit_mapping, modelRanks, modelSuits):
    c = 0

    pred = []

    for rank, suit in rank_suit_mapping:
        # resize the rank and suit to our desired size
        rank = cv2.resize(rank, (constants.CARD_WIDTH, constants.CARD_HEIGHT))
        suit = cv2.resize(suit, (constants.CARD_WIDTH, constants.CARD_HEIGHT))

        # remove any "spots"
        for i in range(rank.shape[0]):
            for j in range(rank.shape[1]):
                if rank[i, j] < 230 and rank[i, j] > 50:
                    rank[i, j] = 255
                if suit[i, j] < 230 and suit[i, j] > 50:
                    suit[i, j] = 255

        cv2.imwrite('{}-0.png'.format(c), rank)
        cv2.imwrite('{}-1.png'.format(c), suit)
        c += 1

        # compare against images
        # bestRank = ""
        # minDiff = 100*120
        # rankDict = dict()
        # for f in os.listdir('imgs/ranks2'):
        #     name = os.path.join('imgs/ranks2', f)
        #     img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        #     diff = int(np.sum(cv2.absdiff(img, rank)/255))
        #
        #     rankName = Path(name).stem.split('-')[0]
        #     if rankName in rankDict:
        #         rankDict[rankName] += diff
        #     else:
        #         rankDict[rankName] = diff
        #
        #     if diff < minDiff:
        #         minDiff = diff
        #         bestRank = Path(name).stem

        # bestSuit = ""
        # minDiff = 100*120
        # suitDict = dict()
        # for f in os.listdir('imgs/suits2'):
        #     name = os.path.join('imgs/suits2', f)
        #     img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
        #     diff = int(np.sum(cv2.absdiff(img, suit)/255))
        #
        #     suitName = Path(name).stem.split('-')[0]
        #     if suitName in suitDict:
        #         suitDict[suitName] += diff
        #     else:
        #         suitDict[suitName] = diff
        #
        #     if diff < minDiff:
        #         minDiff = diff
        #         bestSuit = Path(name).stem

        # get the predictions for suit and rank
        bestSuitPredictions = augtest.model_predict(modelSuits, suit, 'suits')  # min(suitDict, key=suitDict.get)
        bestRankPredictions = augtest.model_predict(modelRanks, rank, 'ranks')  # min(rankDict, key=rankDict.get)

        # get the names and percentage of best and second best suits and ranks
        bestSuitName, bestSuitPer = augtest.model_predictions_to_name(bestSuitPredictions)
        bestRankName, bestRankPer = augtest.model_predictions_to_name(bestRankPredictions)
        sbestSuitName, sbestSuitPer = augtest.model_predictions_to_name(bestSuitPredictions,loc=-2)
        sbestRankName, sbestRankPer = augtest.model_predictions_to_name(bestRankPredictions, loc=-2)

        # show both guesses
        totalPer = bestRankPer + sbestRankPer + bestSuitPer + sbestSuitPer
        guess1 = '{}/{}/{}%'.format(bestRankName, bestSuitName, round(((bestSuitPer + bestRankPer) / totalPer) * 100))
        guess2 = '{}/{}/{}%'.format(sbestRankName, sbestSuitName,
                                    round(((sbestSuitPer + sbestRankPer) / totalPer) * 100))

        pred.append('{}\n{}'.format(guess1, guess2))
        c += 1

    return pred

def show_text(pred, four_corners_set, img):
    for i in range(0, len(pred)):
        # figure out where to place the text
        corners = np.array(four_corners_set[i])
        corners_flat = corners.reshape(-1, corners.shape[-1])
        startX = corners_flat[0][0] + 0
        halfY = (corners_flat[0][1] + corners_flat[1][1]) // 2
        pred_list = pred[i].split('\n')

        font = cv2.FONT_HERSHEY_COMPLEX
        gap = 0
        # show the text
        for j in pred_list:
            cv2.putText(img, j, (startX, halfY + gap), font, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, j, (startX, halfY + gap), font, 0.8, (50, 200, 200), 2, cv2.LINE_AA)
            gap += 30