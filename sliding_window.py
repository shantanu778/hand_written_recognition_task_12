import numpy as np
import os
import tensorflow as tf
import cv2
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
import copy
import matplotlib.pyplot as plt
import argparse
import os

from segmentations.line.line import *

from char_classifier import *
from map_character import *
import pickle



# Here the driving functions are called 
#Specify the stepsize of the sliding window
stepSize = 20
window_width = 25
window_height = 100


# # Sliding window

def sliding_window(image, stepSize, width, height):
    images = []
    y = 0
    for x in range(0, image.shape[1], stepSize):
        if x + width >= 1300:
            return images
        else:
            window = image[y: y+height, x: x+width]
            images.append(window)
    return images

# resize the images for the network
def resize(windows, IMG_WIDTH,IMG_HEIGHT):
    pred_set = []
    dim = (IMG_WIDTH,IMG_HEIGHT)
    for i in range(0,len(windows)):
        window = windows[i]
        image = cv2.resize(window, dim, interpolation= cv2.INTER_AREA)
        pred_set.append(image)
    return pred_set


# Implement a classification threshold
# 'No categorization' 0.20 < 'unsure classification' 0.5 < classification
def reject_weak_pred(prediction):
    values = []
    for i in range (0, len(prediction)):
        value = np.argmax(prediction[i])
        if prediction[i][value] <= 0.2:
            value = 28
        values.append(value)
    return values

def get_unique_order(values):
    uniques = []
    for i in range(0, len(values)-2):
        if values[i] != values[i+1]:
            uniques.append(values[i])
        else:
            continue
    return uniques



def main():
    parser = argparse.ArgumentParser(description='Code for Image Segmentation with Distance Transform and Watershed Algorithm.\
    Sample code showing how to segment overlapping objects using Laplacian filtering, \
    in addition to Watershed and Distance Transformation')
    parser.add_argument('--image_path', help='Path to input image.', default='test_image')
    parser.add_argument('--classifier_path', help='Path to Pretrained Model to Classify Character.', default='cnn.sav')
    parser.add_argument('--flag', help='Model type 0 to 3.', default=0)
    parser.add_argument('--verbose', help='1 or 0 for saving character images.', default=0)
    args = parser.parse_args()

    classifier = args.classifier_path
    loaded_model = pickle.load(open(classifier, 'rb'))

    if args.flag == 0:
        IMG_WIDTH = 60
        IMG_HEIGHT = 40
    else:
        IMG_WIDTH = 100
        IMG_HEIGHT = 100


    dir_path = f"{args.image_path}"
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

    os.makedirs("Text_files/sliding_window", exist_ok=True)
    folder = "Text_files/sliding_window"

    image_names = []

    for file in files:
        imageName = file.split("/")[-1].split(".")[0]
        print(imageName)
        image_names.append(imageName)
        lines, imageName, outputDir  =  segment(file, 'line_images')

        file1 = open(f"{folder}/{imageName}.txt","a")#append mode

        for i in range(len(lines)):
            values = []
            image = lines[i]
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            image = cv2.resize(image, (1300, 100),interpolation = cv2.INTER_AREA)
            image= np.array(image)
            image = image.astype('float32')
            image /= 255 
            print(image.shape)
            windows = sliding_window(image, stepSize, window_width, window_height)
            pred_set = resize(windows, IMG_WIDTH, IMG_HEIGHT)
            prediction = loaded_model.predict(x=np.array(pred_set, np.float32))
            values = reject_weak_pred(prediction)
            uniques = get_unique_order(values)
            
            ascii = map_characters(uniques)

            ascii = [c for c in ascii if c is not None]
            s = str("".join(ascii))
            s = s.replace('#',' ')
            s ="".join(s)
            # print(s)
            file1.write(s)
            file1.write('\n')
            print("complete writing", imageName)
        file1.close()
            

if __name__ == '__main__':
    main()


