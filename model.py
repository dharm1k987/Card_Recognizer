import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from opencv_card_recognizer import augment

def loadData(pathOfDir):
    images = []
    classes = []
    for filename in os.listdir(pathOfDir):
        f = os.path.join(pathOfDir, filename)
        file_alone = filename.split('.')[0]
        file_alone = file_alone.split('-')[0]

        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        list_of_imgs = [
            img,
            augment.brightness_img(img, 30),
            augment.brightness_img(img, -20),
            augment.noise_img(img),
            augment.zoom_img(img, 1.2),
            # augment.horizontal_flip(img),
            augment.blur_image(img),
            augment.rotation(img, 10),
            augment.rotation(img, -10),
            # augment.warpShift(img, 10)
        ]

        c = 0

        for i in list_of_imgs:
            images.append(i)

            if file_alone == 'A':
                classes.append(0)
            elif file_alone == 'J':
                classes.append(10)
            elif file_alone == 'Q':
                classes.append(11)
            elif file_alone == 'K':
                classes.append(12)
            elif file_alone.startswith('Hearts'):
                classes.append(0)
            elif file_alone.startswith('Spades'):
                classes.append(1)
            elif file_alone.startswith('Clubs'):
                classes.append(2)
            elif file_alone.startswith('Diamonds'):
                classes.append(3)
            else:
                classes.append(int(file_alone) - 1)

            # cv2.imwrite('{}-{}.png'.format(file_alone, c), i)
            c += 1

    X = np.array(images)
    y = np.array(classes)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, shuffle=True)

    return X_train, X_test, y_train, y_test


def getModel(img, classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(120,100,1)),

        tf.keras.layers.Conv2D(64, (2,2), activation="relu", padding="same"),
        # tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2,2)),

        tf.keras.layers.Conv2D(128, (2,2), activation="relu", padding="same"),
        # tf.keras.layers.Conv2D(128, (3,3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2,2)),

        tf.keras.layers.Conv2D(256, (2,2), activation="relu", padding="same"),
        # tf.keras.layers.Conv2D(256, (3,3), activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D((2,2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),

        # tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(classes, activation='softmax')

    ])


    model.compile('adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model