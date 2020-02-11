import os
import preprocessing as prep
import model
from keras.models import load_model


def process(self, image_array):
    model = model.load('my_model2.h5')
    digit = prep.crop_image(image_array)
    digit = prep.crop_image(digit)
    digit = prep.center_image(digit)
    digit = prep.resize_image(digit)
    digit = prep.min_max_scaler(digit)
    digit = prep.reshape_array(digit)
    digit = model.predict(digit)
    return digit

