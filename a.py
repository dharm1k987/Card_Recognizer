import os
import cv2
import numpy as np
from pathlib import Path



for f in os.listdir('imgs/ranks3'):
    filename = os.path.join('imgs/ranks3', f)
    print(filename)
    img = cv2.imread(filename)
    sharpened = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('or', sharpened)
    justChanged = set()
    for i in range(sharpened.shape[0]):
        for j in range(sharpened.shape[1]):
            if sharpened[i,j] < 50 and (i,j) not in justChanged:
                change = 3
                # sharpened[i,j] = 65#max(0, sharpened[i,j] - 40)
                if j - change > 0:
                    sharpened[i,j-change] = sharpened[i,j]
                if j + change < 100:
                    sharpened[i,j+change ] = sharpened[i,j]
                if i + change < 120:
                    sharpened[i+change,j] = sharpened[i,j]
                if i - change > 0:
                    sharpened[i-change,j] = sharpened[i,j]

                if i + change < 120 and j + change < 100:
                    sharpened[i+change,j+change] = sharpened[i,j]

                if i + change < 120 and j - change > 0:
                    sharpened[i+change,j-change] = sharpened[i,j]

                if i - change > 0 and j - change > 0:
                    sharpened[i-change,j-change] = sharpened[i,j]

                if i - change > 0 and j + change < 100:
                    sharpened[i-change,j+change] = sharpened[i,j]


                justChanged.add((i,j-change))
                justChanged.add((i,change+j))
                justChanged.add((i+change,j))
                justChanged.add((i-change,j))
                justChanged.add((i+change,j+change))
                justChanged.add((i+change,j-change))
                justChanged.add((i-change,j-change))
                justChanged.add((i-change,j+change))

                # sharpened[i+5,j+5] = 0






    # if sharpened[i,j] > 150:
    #     sharpened[i,j] = 255#min(255, sharpened[i,j] + 40)


    name = Path(filename).stem
    num = name.split('-')
    if len(num) > 1:
        num = int(num[1])
    else:
        num = 0

    num += 192

    cv2.imwrite('imgs/ranks6/{}-{}.png'.format(name.split('-')[0], num), sharpened)
    # cv2.waitKey(0)

    # break
