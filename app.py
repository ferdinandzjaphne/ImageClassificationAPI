from flask import Flask, request, jsonify
import json 
app = Flask(__name__)
import numpy as np
import os
import cv2
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import base64
from keras.models import load_model
from PIL import Image
from io import BytesIO

def predict_image_class(image):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(100, 100, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5))
    model.add(Activation('sigmoid'))



    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

    model.load_weights('50_epochs.h5')


    image = image.resize((100,100))

    x = img_to_array(image)

    x = x.reshape((1,) + x.shape)


    classes = model.predict_classes(x)

    prediction_result = classes.item(0)


    if prediction_result == 0:
        class_name = "hydro coco"
    elif prediction_result == 1:
        class_name = "mintz"
    elif prediction_result == 2:
        class_name = "morisca"
    elif prediction_result == 3:
        class_name = "pop mie"
    elif prediction_result == 4:
        class_name = "ultra milk" 

    return class_name

@app.route('/', methods=['GET'])
def return_response():
    class Response:
        def __init__(self, product1, product2):
            self.product1 = product1
            self.product2 = product2
            
        def __str__(self):
            return self.status


    new_response = Response(
        product1 = "oreo",
        product2 = "fruit tea"
    )

    def convert_to_dict(obj):
        """
        A function takes in a custom object and returns a dictionary representation of the object.
        This dict representation includes meta data such as the object's module and class names.
        """
        
        #  Populate the dictionary with object meta data 
        obj_dict = {
            "__class__": obj.__class__.__name__,
            "__module__": obj.__module__
        }
        
        #  Populate the dictionary with object properties
        obj_dict.update(obj.__dict__)
        
        return obj_dict
    

    data = json.dumps(new_response, default=convert_to_dict, indent=4, sort_keys=True)
    return data
    

@app.route('/', methods=['POST'])
def parse_req():
    json_data = request.get_json(force=True)

    
    testStringConstruct = json_data['testString']

    class Response:
        def __init__(self, testString):
            self.testString = testString
            
        def __str__(self):
            return self.status


    new_response = Response(
        testString = testStringConstruct
    )

    def convert_to_dict(obj):
        """
        A function takes in a custom object and returns a dictionary representation of the object.
        This dict representation includes meta data such as the object's module and class names.
        """
        
        #  Populate the dictionary with object meta data 
        obj_dict = {
            "__class__": obj.__class__.__name__,
            "__module__": obj.__module__
        }
        
        #  Populate the dictionary with object properties
        obj_dict.update(obj.__dict__)
        
        return obj_dict
    

    data = json.dumps(new_response, default=convert_to_dict, indent=4, sort_keys=True)
    return data


# @app.route('/', methods=['POST'])
# def return_class():
#     json_data = request.get_json(force=True)

#     encodedImage = json_data['testString']

#     try:
#         image = Image.open(BytesIO(base64.b64decode(encodedImage)))
#     except Exception as e :
#         encodedImage = str(e)


#     image_class = predict_image_class(image)

#     class Response:
#         def __init__(self, testString):
#             self.testString = testString
            
#         def __str__(self):
#             return self.status


#     new_response = Response(
#         testString = image_class
#     )

#     def convert_to_dict(obj):
#         """
#         A function takes in a custom object and returns a dictionary representation of the object.
#         This dict representation includes meta data such as the object's module and class names.
#         """
        
#         #  Populate the dictionary with object meta data 
#         obj_dict = {
#             "__class__": obj.__class__.__name__,
#             "__module__": obj.__module__
#         }
        
#         #  Populate the dictionary with object properties
#         obj_dict.update(obj.__dict__)
        
#         return obj_dict
    

#     data = json.dumps(new_response, default=convert_to_dict, indent=4, sort_keys=True)
#     return data


if __name__ =='__main__':
    app.run()