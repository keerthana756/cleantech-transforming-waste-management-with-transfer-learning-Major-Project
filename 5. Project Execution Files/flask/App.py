from flask import Flask, render_template, request, jsonify, redirect, url_for
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import tensorflow as tf
import os

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model(r'C:\Users\velup\OneDrive\Desktop\Majorproject\vgg16.h5')

# Ensure the uploads folder exists
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Home route
@app.route('/')
def index():
    return render_template("index.html")

# About route
@app.route('/about')
def about():
    return render_template("about.html")

# Contact route
@app.route('/contact')
def contact():
    return render_template("contact.html")

# Prediction route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'POST':
            # Check if the file part is present in the request
            if 'pc_image' not in request.files:
                return render_template("predict.html", predict="No file part")
            
            f = request.files['pc_image']
            
            # Check if the user selected a file
            if f.filename == '':
                return render_template("predict.html", predict="No file selected")

            # Save the uploaded image
            img_path = os.path.join(UPLOAD_FOLDER, f.filename)
            f.save(img_path)

            # Load and preprocess the image
            img = load_img(img_path, target_size=(224, 224))  # Ensure correct image size
            image_array = np.array(img)  # Convert the image to a numpy array
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

            # Make prediction
            prediction = model.predict(image_array)  # Get model's prediction probabilities
            pred_class = np.argmax(prediction, axis=1)  # Get predicted class index
            
            # Map the predicted class index to a category
            index = ['Biodegradable Image (0)', 'Recyclable Image (1)', 'Trash Image(2)']
            result = index[int(pred_class[0])]  # Extract the scalar value from the array

            # Return prediction result to the template
            return render_template("predict.html", predict=result)
        else:
            # If it's a GET request, just render the empty predict page
            return render_template("predict.html")
    
    except Exception as e:
        # In case of an error, display an error message
        return render_template("predict.html", predict=f"An error occurred: {e}")

if __name__ == '__main__':
    # Run the app
    app.run(debug=True, port=2222)

