import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, ZeroPadding2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import cv2
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print("Using GPU")
else:
    print("Using CPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.models import load_model
model = load_model('models/cat_dog_model.h5')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgRGB_resize = cv2.resize(imgRGB, (224, 224))
    imgRGB_process = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).standardize(imgRGB_resize)
    imgRGB_process = np.expand_dims(imgRGB_process, axis=0)

    # print(imgRGB_process)
    # print(imgRGB_process.shape)
    # print(type(imgRGB_process[0][0][0][0]))
    predictions = model(imgRGB_process).numpy()
#    np.round(predictions)
    print(predictions)

    predictions_round = np.round(predictions)

    class_label = ["CAT","DOG"]
    detect_label = class_label[np.argmax(predictions_round==1)]
    detect_confidence = predictions[0][np.argmax(predictions_round==1)]
    detect_confidence_percent = str(round(detect_confidence*100,2))
    print(detect_confidence)

    if np.amax(predictions) > 0.95:
        cv2.putText(img, detect_label, (250,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 5, cv2.LINE_AA)
        cv2.putText(img, detect_label, (250, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(img, detect_confidence_percent, (250,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 5, cv2.LINE_AA)
        cv2.putText(img, detect_confidence_percent, (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Image", img)

    c = cv2.waitKey(1)
    if c == 27:
        break

cv2.release()
cv2.destroyAllWindows()
