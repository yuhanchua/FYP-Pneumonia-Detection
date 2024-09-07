from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import io  # Import io for BytesIO
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the trained model
model = load_model('pneumonia_detection_model.h5', compile=False)

# Define a route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    try:
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess the uploaded image (convert file stream to BytesIO)
        img = image.load_img(file_path, target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.0

        # Make prediction
        prediction = model.predict(img)
        output = 'Pneumonia Detected' if prediction[0][0] > 0.5 else 'Normal'
        
        # Pass the result and image path to the result page
        image_url = url_for('static', filename=f'uploads/{filename}')
        return render_template('result.html', result=output, image_url=image_url)
    
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
