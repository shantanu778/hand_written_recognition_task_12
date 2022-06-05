from fileinput import filename
import pandas as pd
import numpy as np
import random
import os
import tensorflow as tf
import cv2
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg


img_folder = 'DSS'


def create_dataset(img_list, IMG_WIDTH, IMG_HEIGHT):
    img_data_array=[]
    for img in img_list: 
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # print(img.shape)
        image= cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT),interpolation = cv2.INTER_AREA)
        # print(image.shape)
        image= np.array(image)
        #image = image.reshape((40, 60, -1))
        #print(image.shape)
        image = image.astype('float32')
        image /= 255 
        img_data_array.append(image)
    return img_data_array



def predict(img_data, loaded_model):
    # loaded_model.summary()
    # Predict classifications
    prediction = loaded_model.predict(x=np.array(img_data, np.float32))
    return prediction



def target_dic(class_name):
    target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
    target_val =  [target_dict[class_name[i]] for i in range(len(class_name))]
    y = list(map(int,target_val))
    target_dict["None"] = 28

    return target_dict, target_val

##Translate the precition into a word
def get_keys_from_value(d, val):
    for k, v in d.items():
        if v == val:
            return k


def char_word(prediction, img_data, target_dict):
    word = []
    values = []
    letters = []

    for i in range (0, len(img_data)):
        value = np.argmax(prediction[i])
        # print(value)
        if prediction[i][value] <= 0.2:
            value = 28
        letter = get_keys_from_value(target_dict, value)
        letters.append(letter)
        values.append(value)
    return letters, values
    
    


def classify(img_list, loaded_model, IMG_WIDTH, IMG_HEIGHT):
    img_data = create_dataset(img_list,  IMG_WIDTH, IMG_HEIGHT)
    predictions = predict(img_data, loaded_model)

    class_name = []
    for dir in os.listdir(img_folder): 
        class_name.append(dir) 

    target_dict, target_val = target_dic(class_name)

    letters, values = char_word(predictions, img_data, target_dict)

    # print("These are the classification output values for the specified line image:")
    # print(letters)

    return letters, values

                                     

