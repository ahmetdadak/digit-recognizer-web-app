from flask import Flask, render_template
from flask import request
import numpy as np
from PIL import Image
import base64
from io import BytesIO     
from io import StringIO
from PIL import Image
from flask import jsonify
import preprocessing as prep 
from keras.models import load_model

import pandas as pd
import numpy as np
import os

import itertools

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/foo', methods=['POST'])  
def foo():
    if request.method == "POST":
        K.clear_session()

        data = request.form["data"]
        encoded_data = data.split(',')[1]
        img = base64.b64decode(encoded_data)
        filename = 's.png'  
        with open(filename, 'wb') as f:
                f.write(img)
        img = Image.open(filename).convert('L')
        nparray = np.array(img)
        nparray2 = np.abs(255-nparray)
        number = process(nparray2)
        
        return jsonify(result=number)

def process(image_array):
    model = load_model('my_model2.h5')
    digit = prep.crop_image(image_array)
    digit = prep.crop_image(digit)
    digit = prep.center_image(digit)
    digit = prep.resize_image(digit)
    digit = prep.min_max_scaler(digit)
    digit = prep.reshape_array(digit)
    digit = model.predict(digit)
    i = "Draw Again!"
    predict = False
    count = 0
    for number in digit[0]:
        if np.max(digit[0]) == number and number>0.70:
            print(number)
            predict = True
            break
        else:
            count = count + 1
    if predict == True:
        return count
    else:
        return i



@app.route('/display', methods=['POST'])  
def display():
    number = 1
    return jsonify(result=number )


if __name__ == "__main__":
    app.run(debug=True)