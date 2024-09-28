# app.py

import os
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import uuid

# Initialize the Flask app
app = Flask(__name__)

# Set the folder for uploads
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the trained model
model = load_model('best_model.keras')

# Class labels (replace these with your actual labels)
class_labels = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
                'Potato___Late_blight', 'Potato___healthy', 'Tomato__Bacterial_spot',
                'Tomato__Early_blight', 'Tomato__Late_blight', 'Tomato__Leaf_Mold',
                'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites_Two_spotted_spider_mite',
                'Tomato__Target_Spot', 'Tomato__Tomato_Yellow_Leaf_Curl_Virus', 
                'Tomato__Tomato_mosaic_virus', 'Tomato__healthy']

# Helper function to check if the uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file is submitted
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # If no file selected, redirect back to the page
        if file.filename == '':
            return redirect(request.url)

        # If file is allowed, save it and process
        if file and allowed_file(file.filename):
            # Generate a unique filename for each uploaded image
            filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess the image for the model
            img = Image.open(filepath)
            img = img.resize((224, 224))  # Resize to the input size expected by the model
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Scale the image

            # Perform prediction
            prediction = model.predict(img_array)
            predicted_class = class_labels[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            return render_template('result.html', 
                                   filename=filename, 
                                   prediction=predicted_class, 
                                   confidence=round(confidence, 2))

    return render_template('index.html')


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
