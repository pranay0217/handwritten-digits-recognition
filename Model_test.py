import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('newModel.keras')
image_number = 1
while os.path.isfile('digits/digit{}.png'.format(image_number)):
    try:
        img = cv2.imread('digits/digit{}.png'.format(image_number))[:,:,0]
        img = np.invert(np.array([img]))
        img = img[0]                   # shape (28, 28)
        img = img.reshape(1, 784)         # reshape to (1, 784)
        prediction = model.predict(img)
        print("The number is probably a {}".format(np.argmax(prediction)))
        image_number += 1
    except Exception as e:
        # print("Error reading image! Proceeding with next image... Error: {}".format(e))
        image_number += 1