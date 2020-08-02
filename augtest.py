import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import Callback
import numpy as np
from opencv_card_recognizer import model

def model_predict(rankOrSuitModel, data, type):
    data = data.reshape(data.shape[0], data.shape[1], 1)
    data = np.expand_dims(data, axis=0)
    pred = rankOrSuitModel.predict(  np.vstack([data])    )[0]

    # d = []
    # if type == 'suits':
    #     d.extend(['Hearts', 'Spades', 'Clubs', 'Diamonds'])
    # else:
    #     d.extend(['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'])

    return pred#d[np.argmax(pred, axis=0)]

def model_wrapper(dataPath, classes, wtsPath=None, train=False, toSaveAs=None):     # 'imgs/ranks'

    X_train, X_test, y_train, y_test = model.loadData(dataPath)

    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)

    dataGen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.4,
        height_shift_range=0.4,
        shear_range = 0.2,
        zoom_range=0.2,
        fill_mode='nearest',
        horizontal_flip=False,
        channel_shift_range=0.1,
        rescale=1.1
    )
    dataGen.fit(X_train)

    myModel = model.getModel(X_train[0], classes)

    y_train = to_categorical(y_train,classes)
    y_test = to_categorical(y_test,classes)

    if wtsPath:
        myModel.load_weights(wtsPath)

    if train:
        class myCallback(Callback):
            def on_epoch_end(self, epoch, logs={}):
                if logs.get('accuracy') > 0.90 and logs.get('val_accuracy') > 0.90:
                    print('Stopping training')
                    myModel.stop_training = True

        history = myModel.fit_generator(dataGen.flow(X_train,y_train,
                        batch_size=8),
                        steps_per_epoch=len(X_train)//8,
                        epochs=2000,
                        validation_data=(X_test,y_test),
                        shuffle=1,
                        callbacks=[myCallback()])

        if wtsPath:
            myModel.save_weights(wtsPath)
        else:
            myModel.save_weights(toSaveAs)

        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.legend(['training','validation'])
        plt.title('Loss')
        plt.xlabel('epoch')
        plt.figure(2)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.legend(['training','validation'])
        plt.title('Accuracy')
        plt.xlabel('epoch')
        plt.show()

    return myModel
#
# myModel = model_wrapper('imgs/ranks2',13, None, train=True, toSaveAs='ranksOvernightWeights.h5')
# myModel2 = model_wrapper('imgs/ranks2',13, None, train=True, toSaveAs='ranksOvernightWeights2.h5')