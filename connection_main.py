from flask import Flask, request, jsonify
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image 
from io import BytesIO
from flask import Flask, request, jsonify
import base64
from pymongo import MongoClient

model = tf.keras.models.load_model('D:/Project/DR/model_alexnet_wiener_clahe_g.h5')
# Initialize the Flask application
app = Flask(__name__)
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

def preprocess_image(image_bytes):
    img =Image.open(BytesIO(image_bytes)).conert('RGB')
    img_array = np.array(img)
    img_array = img_array/255
    img_array =np.expand_dims(img_array, axis=0)
    return img_array

def predict(image_bytes):
    img_array = preprocess_image(image_bytes)
    predictions =model,predict(img_array)
    predicted_class=np.argmax(predictions[0])
    return predicted_class

def readb64(base64_string):
    sbuf = BytesIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf).convert('RGB')
    return pimg
def process_image_and_predict(base64_string):
    image = readb64(base64_string)
    image_bytes = cv2.imencode('.jpg', np.array(image))[1].tobytes()
    predicted_class = predict(image_bytes)
    return predicted_class

# Generate predictions
#y_preds = model.predict(img_array)

# Print the predicted class probabilities
#print(y_preds)

# class_code = {0: "No_DR",
#               1: "Mild", 
#               2: "Moderate",
#               3: "Severe",
#               4: "Proliferate_DR"}
# predicted_class = class_code[y_preds.argmax()]
# print("Predicted class:", predicted_class)

@app.route('/api/upload', methods=['POST'])
def index():
    if request.method == 'POST':
        # Get the user details from the request
        name = request.form['name']
        phone = request.form['phone']
        email = request.form['email']
        # Get the image data from the request
        image = request.files['file'].read()
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
    app.run(port=500,debug=True)

app = Flask(__name__)
client = MongoClient('mongodb://localhost:27017/')
db = client['image_database']