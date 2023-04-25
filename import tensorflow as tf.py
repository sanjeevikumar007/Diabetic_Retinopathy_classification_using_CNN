from flask import Flask, request, render_template
from pymongo import MongoClient
from PIL import Image
import base64
import io
import numpy as np
import tensorflow as tf

app = Flask(__name__)
client = MongoClient('mongodb://localhost:27017/')
db = client['image_database']

# Load the ML model
model = tf.keras.models.load_model('model.h5')

# Define a function to preprocess the image data for the model
def preprocess_image(image):
    # Resize the image to the input shape of the model
    image = image.resize((224, 224))
    # Convert the image to a numpy array
    image = np.array(image)
    # Normalize the pixel values to be between 0 and 1
    image = image / 255.0
    # Add an extra dimension to represent the batch size
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the user details from the request
        name = request.form['name']
        phone = request.form['phone']
        email = request.form['email']
        # Get the image data from the request
        image = request.files['image'].read()
        # Convert the image to a PIL Image object
        image = Image.open(io.BytesIO(image))
        # Preprocess the image data for the model
        image = preprocess_image(image)
        # Run the image through the model to get the prediction
        prediction = model.predict(image)[0]
        # Store the user details, image, and prediction in the database
        encoded_image = base64.b64encode(image).decode('utf-8')
        db.images.insert_one({'name': name, 'phone': phone, 'email': email, 'image': encoded_image, 'prediction': float(prediction)})
        # Display the prediction on the webpage
        return render_template('result.html', prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
