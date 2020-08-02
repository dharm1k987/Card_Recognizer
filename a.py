import os
import cv2
import numpy as np
from pathlib import Path



for f in os.listdir('imgs/suits3'):
    filename = os.path.join('imgs/suits3', f)
    img = cv2.imread(filename)
    sharpened = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(sharpened.shape[0]):
        for j in range(sharpened.shape[1]):
            if sharpened[i,j] < 150:
                sharpened[i,j] = 65#max(0, sharpened[i,j] - 40)
            if sharpened[i,j] > 150:
                sharpened[i,j] = 255#min(255, sharpened[i,j] + 40)


    name = Path(filename).stem
    num = name.split('-')
    if len(num) > 1:
        num = int(num[1])
    else:
        num = 0

    num += 32

    cv2.imwrite('imgs/suits3/{}-{}.png'.format(name.split('-')[0], num), sharpened)

