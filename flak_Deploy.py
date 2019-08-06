# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import keras

# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from keras.models import Sequential
import cv2
from keras.models import load_model
# Keras

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import tensorflow as tf
from keras.backend import clear_session
# Define a flask app
app = Flask(__name__)


CATEGORIES=['Dog','Cat']
# Load your trained model
global model
model = load_model("63x3-CNN.model")
graph = tf.get_default_graph()



print('Model loaded. Check http://127.0.0.1:5000/')





@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

def prepare(filepath):
    image_size=60
    
    img_array=cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    new_array=cv2.resize(img_array,(image_size,image_size))
    return new_array.reshape(-1,image_size,image_size,1)    

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
       
        # Get the file from post request
        f = request.files['file']
        
        
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print(file_path)
        with graph.as_default():
       
            prediction=model.predict(prepare(file_path))  
            
            result = CATEGORIES[int(prediction[0][0])]  
            
            return result 
    return None        # Convert to string

      # Make prediction
        
        #preds = model_predict(file_path, model)
        #prediction_=model.predict([prepare('4.jpg')])
        #return = CATEGORIES[int(prediction_[0][0])])
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=4)
        # ImageNet Decode
       
     


if __name__ == '__main__':
    app.run(debug=True)
