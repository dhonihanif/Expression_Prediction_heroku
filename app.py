from http.client import REQUEST_HEADER_FIELDS_TOO_LARGE
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow as tf
import re
from flask import Flask, render_template, url_for, request
import cv2
from keras.models import load_model
from PIL import Image
<<<<<<< HEAD
import base64
import io
=======
from tensorflow.keras.preprocessing import image
>>>>>>> c2dbe869 (new)

import cv2

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = load_model("./expression_detection/facial_emotions_model.h5")
model.compile(loss="binary_crossentropy",
optimizer="rmsprop", metrics=["accuracy"])
classes=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

app = Flask(__name__, template_folder="templates")
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def hello_world():
    return render_template("index.html")

@app.route("/post/predict", methods=["GET", "POST"])
def detect_faces():
    if request.method == "POST":

<<<<<<< HEAD
        img = request.files.get("file", "")
        img = request.form["file"]
    # test_image=image.load_img(img_path,target_size=(48,48),color_mode='grayscale')
    # test_image=image.img_to_array(test_image)
    # test_image=test_image.reshape(1,48,48,1)
    # classes=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
    # result=model.predict(test_image)
    # y_pred=np.argmax(result[0])
    # print('The person facial emotion is:',classes[y_pred])
=======
    return render_template("./index.html")

@app.route("/post/predict", methods=["GET", "POST"])
def detect_faces():
    test_image = request.files["image"]
    
    test_image=image.load_img(test_image,target_size=(48,48),color_mode='grayscale')
    test_image=image.img_to_array(test_image)

    test_image = test_image.reshape(1,48,48,1)
    classes = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
    result = model.predict(test_image)
    print(result[0])
    y_pred = np.argmax(result[0])
    print('The person facial emotion is:',classes[y_pred])
>>>>>>> c2dbe869 (new)
    return render_template("halaman1.html")

if __name__ == "__main__":
    app.run(debug=True)