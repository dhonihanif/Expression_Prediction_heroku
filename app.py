from unittest import result
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
import re
from flask import Flask, render_template, url_for, request, flash, redirect
from keras.models import load_model
from PIL import Image, ImageOps
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

import io
import os

app = Flask(__name__)                                    #calling
def get_model():
    global model
    model = load_model('./expression_detection/facial_emotions_model.h5')
    print("Model loaded!")

def load_image(img_path):

    img = load_img(img_path, target_size=(48, 48), color_mode='grayscale')
    img_tensor = img_to_array(img)                    # (height, width, channels)
    img_tensor= img_tensor.reshape(1,48,48,1)
        
    return img_tensor

get_model()
def prediction(img_path):
    new_image = load_image(img_path)
    
    result = model.predict(new_image)
    classes=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
    y_pred=np.argmax(result[0])
    return classes[y_pred]


@app.route("/", methods=['GET', 'POST'])                 #initialising
def home():                                              #function call

    return render_template('index.html')                  #return and calling HTML page (designed template)


@app.route("/predict", methods = ['GET','POST'])
def predict():
    
    if request.method == 'POST':
        
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join("", filename)                       #slashes should be handeled properly
        file.save(file_path)
        product = prediction(file_path)
        
        if product == "Angry":
            url = "https://open.spotify.com/search/angry"
        elif product == "Disgust":
            url = "https://open.spotify.com/search/disgust"
        elif product == "Fear":
            url = "https://open.spotify.com/search/fear"
        elif product == "Happy":
            url = "https://open.spotify.com/search/happy"
        elif product == "Neutral":
            url = "https://open.spotify.com/search/neutral"
        elif product == "Sad":
            url = "https://open.spotify.com/search/sad"
        elif product == "Surprise":
            url = "https://open.spotify.com/search/surprise"
    
    return render_template('predict.html', product = product, user_image = file_path, url=url)    
if __name__ == "__main__":
    app.run(port=5000)