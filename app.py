from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import logging
from prediction import segment_and_recognize_letters

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # get image from request
        img = request.files['img'].stream
        img = cv2.imdecode(np.fromstring(img.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # make prediction
        segmented_letters = segment_and_recognize_letters(img)

        response = {
            str(segmented_letters),
            
        }
        
        return jsonify(response)