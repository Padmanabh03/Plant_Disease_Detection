import os
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import uuid

# Initialize the Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')  # Ensure path is correct
MODEL_FOLDER = 'models'  # Directory where your model is stored
MODEL_NAME = 'best_model.keras'  # Model filename

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

# Allowed file extensions for uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the trained model once when the app starts
model_path = os.path.join(MODEL_FOLDER, MODEL_NAME)
model = load_model(model_path)

# Class labels (replace these with your actual labels)
class_labels = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Tomato__Bacterial_spot',
    'Tomato__Early_blight', 'Tomato__Late_blight', 'Tomato__Leaf_Mold',
    'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato__Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato__Tomato_mosaic_virus', 'Tomato__healthy'
]

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            error_message = 'No file part in the request.'
            return render_template('index.html', error=error_message)

        file = request.files['file']

        # If no file is selected
        if file.filename == '':
            error_message = 'No file selected for uploading.'
            return render_template('index.html', error=error_message)

        # If the file is allowed, process it
        if file and allowed_file(file.filename):
            try:
                # Generate a unique filename using UUID
                unique_filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)

                # Preprocess the image for the model
                img = Image.open(filepath).convert('RGB')  # Ensure image is in RGB
                img = img.resize((224, 224))  # Resize as per model's requirement
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0  # Normalize the image

                # Perform prediction
                prediction = model.predict(img_array)
                predicted_class = class_labels[np.argmax(prediction)]
                confidence = np.max(prediction) * 100

                return render_template('result.html', 
                                       filename=unique_filename, 
                                       prediction=predicted_class, 
                                       confidence=round(confidence, 2))
            except Exception as e:
                # Handle exceptions (e.g., invalid image format)
                error_message = f"An error occurred while processing the image: {str(e)}"
                return render_template('index.html', error=error_message)

        else:
            error_message = 'Allowed file types are png, jpg, jpeg.'
            return render_template('index.html', error=error_message)

    return render_template('index.html')

@app.route('/display/<filename>')
def display_image(filename):
    """Route to display the uploaded image."""
    # Securely join the filename to prevent directory traversal
    return redirect(url_for('static', filename=os.path.join('uploads', filename)), code=301)

if __name__ == "__main__":
    # Bind to PORT if defined, otherwise default to 5000 (useful for local testing)
    port = int(os.environ.get('PORT', 5000))
    # Disable debug mode for production
    app.run(host='0.0.0.0', port=port, debug=False)
