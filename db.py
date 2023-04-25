import pymongo
import bson.binary
from PIL import Image
from datetime import datetime

# establish a connection to the MongoDB server
client = pymongo.MongoClient("mongodb://localhost:27017")

# create a new database
db = client["mydatabase"]

# generate a unique collection name based on the current date and time
collection_name = "mycollection_" + datetime.now().strftime("%Y%m%d_%H%M%S")

# create a new collection with the unique name
db.create_collection(collection_name)
collection = db.get_collection(collection_name)



# open left eye image file and convert to binary data
with open("D:/Project/BioHack/dataset/eyepacs_preprocess/eyepacs_preprocess/10_left.jpeg", "rb") as f:
    eye_binary = bson.binary.Binary(f.read())

# open right eye image file and convert to binary data


# create document object with fields and binary data
document = {
    "username": "hari",
    "phone_number": "7094337618",
    "email": "hari@gmail.com",
    "image_eye": eye_binary,
    "prediction":"Proliferate"
    
}

# insert document into MongoDB collection
collection.insert_one(document)