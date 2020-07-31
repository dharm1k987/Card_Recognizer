import cv2
import numpy as np
from scipy.ndimage import zoom

def brightness_img(img, value):
    # convert to HSV, add the value in the value channel, and conv back to grya
    test = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(test, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def contrast(img, value):
    brightness = 40
    shadow = brightness
    highlight = 255

    alpha_b = (highlight - shadow)/255
    gamma_b = shadow

    img = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)


    f = 131*(value + 127)/(127*(131-value))
    alpha_c = f
    gamma_c = 127*(1-f)

    img = cv2.addWeighted(img, alpha_c, img, 0, gamma_c)

    return img



def zoom_img(img, zoom_factor):
    h, w = img.shape[:2]
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)
    zh = int(np.round(h / zoom_factor))
    zw = int(np.round(w / zoom_factor))
    top = (h - zh) // 2
    left = (w - zw) // 2

    out = zoom(img[top:top + zh, left:left + zw], zoom_tuple)

    # `out` might still be slightly larger than `img` due to rounding, so
    # trim off any extra pixels at the edges
    trim_top = ((out.shape[0] - h) // 2)
    trim_left = ((out.shape[1] - w) // 2)
    out = out[trim_top:trim_top + h, trim_left:trim_left + w]

    return out


def horizontal_flip(img):
    return np.fliplr(img)


def noise_img(img):
    gaussian = np.random.normal(0, 20, (img.shape[0], img.shape[1]))
    return img + gaussian

def blur_image(img):
    return cv2.GaussianBlur(img, (11, 11), 0)

def rotation(img, angle):
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    result = zoom_img(result, 1.2)
    return result
