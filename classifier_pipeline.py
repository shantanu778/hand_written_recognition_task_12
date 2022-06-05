import pandas as pd
import argparse
import numpy as np
import os
import tensorflow as tf
import cv2
import random
import imgaug as ia
import imgaug.augmenters as iaa
from tensorflow.keras.layers import  Input
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.applications import inception_v3      #flag = 1
from tensorflow.keras.applications import ResNet50          #flag = 2 
from tensorflow.keras.applications import InceptionResNetV2 #flag = 3

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', help='Path to Save the Models.', default='Models')
parser.add_argument('--epochs', help='Number of Epochs', default=50)
parser.add_argument('--aug_no', help='Number of Augmentations per class', default=0)
parser.add_argument('--flag', help='Model type 0 to 3.', default=0)
args = parser.parse_args()

img_folder=r'DSS'

#parameters to be chosen 
#flag = 0 is the implemented CNN
flag = int(args.flag)
n_epochs = int(args.epochs)
number_of_augmentations_per_class = int(args.aug_no) #if set to 0 -  no data augmentation

#some transfer learning models are trained on larger image width and height than in our data set and do not work 
#on smaller sizes
if flag == 0:
    IMG_WIDTH = 40
    IMG_HEIGHT = 60
else:
    IMG_WIDTH = 100
    IMG_HEIGHT = 100

def create_dataset(img_folder,IMG_WIDTH,IMG_HEIGHT):
    img_data_array = []
    class_name = []
    for dir1 in os.listdir(img_folder): 
        letter_path = os.path.join(img_folder, dir1)
        if letter_path != 'monkbrill/.DS_Store':
            for file in os.listdir(letter_path):
                image_path = os.path.join(img_folder, dir1,  file)
                image = cv2.imread(image_path)
                image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
                image = np.array(image)
                image = image.astype('float32')
                image /= 255 
                img_data_array.append(image)
                class_name.append(dir1)
    return img_data_array, class_name


# Three-way (training, validation and testing) split
def split(img_data, class_name):
    target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
    target_val =  [target_dict[class_name[i]] for i in range(len(class_name))]
    y = list(map(int,target_val))
    
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10

    # train is now 75% of the entire data set
    x_train, x_test, y_train, y_test = train_test_split(img_data, y, test_size = 1 - train_ratio, random_state = 42,  stratify = y)

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size = test_ratio/(test_ratio + validation_ratio), random_state = 42,
                                                    stratify = y_test)

    return x_train, x_test, y_train, y_test, x_val, y_val 


# Data Augmentation
def rotating_left (image):
    rotate = iaa.Affine(rotate=(-15, 15))
    rotated_image = rotate.augment_image(image)
    return rotated_image

def rotating_right (image):
    rotate = iaa.Affine(rotate = (15, -15))
    rotated_image = rotate.augment_image(image)
    return rotated_image

def noise_add_gaussian (image):
    gaussian_noise = iaa.AdditiveGaussianNoise(1,1)
    noise_image = gaussian_noise.augment_image(image)
    return noise_image
    
def shear_transform_right (image):
    shear = iaa.Affine(shear = (0,40))
    shear_image = shear.augment_image(image)
    return shear_image

def shear_transform_left (image):
    shear = iaa.Affine(shear = (40,0))
    shear_image = shear.augment_image(image)
    return shear_image

def cropping (image):
    crop = iaa.Crop(percent = (0, 0.3)) # crop image
    corp_image = crop.augment_image(image)
    return corp_image

def blurring (image):
    blurred = iaa.GaussianBlur(sigma=(0.0, 3.0))
    blurred_image = blurred.augment_image(image)
    return blurred_image

def append_augmented_data(augmented_df, augmented_image, augmented_label):
    new_row = {'image': augmented_image, 'label':augmented_label}
    augmented_df = augmented_df.append(new_row, ignore_index = True)
    return augmented_df

def create_data_frame(X_train , y_train):
    train = {'image': X_train, 'label': y_train}
    training_data = pd.DataFrame(data = train)
    return training_data

def data_augmentation (augmented_df):
    for label_id in range(0,27):
        j = 0
        while(j < number_of_augmentations_per_class):
            to_be_augmented_row = augmented_df.sample()
            image = to_be_augmented_row['image']
            image = np.array(image)
            image = image[0]

            a = random.randint(0, 6)
            if a == 0:
                blurred = blurring (image)
                augmented_df = append_augmented_data(augmented_df,blurred,label_id)
                
            if a == 1:
                rotated_image_right = rotating_right(image)
                augmented_df = append_augmented_data(augmented_df,rotated_image_right,label_id)
                
            if a == 2:
                rotated_image_left = rotating_left(image)
                augmented_df = append_augmented_data(augmented_df,rotated_image_left,label_id)
                
            if a == 3:
                image_noise = noise_add_gaussian(image)
                augmented_df = append_augmented_data(augmented_df,image_noise,label_id) 
                
            if a == 4:
                shear_right = shear_transform_right(image)
                augmented_df = append_augmented_data(augmented_df,shear_right,label_id) 
                
            if a == 5:
                shear_left = shear_transform_left(image)
                augmented_df = append_augmented_data(augmented_df,shear_left,label_id) 
                
            if a == 6:
                cropped_image = cropping (image)
                augmented_df = append_augmented_data(augmented_df,cropped_image,label_id)

            j = j+1

    #formating
    x_train = augmented_df.drop(columns=['label'])
    y_train = augmented_df.drop(columns=['image'])

    x_train = x_train.values.tolist()
    y_train = y_train.values.tolist()

    x_train = np.array(x_train, np.float32)
    y_train = np.array(y_train, np.float32)
    x_train = x_train[:, 0, :, :]

    return x_train, y_train


# Deep learning models
def train(flag, IMG_WIDTH, IMG_HEIGHT, X_train, y_train, X_val, y_val):
    if flag == 0:
        model=tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(IMG_WIDTH,IMG_HEIGHT, 3)),
                    tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)), 
                    tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                    tf.keras.layers.Dropout(.25, input_shape=(2,)),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(27),
                    tf.keras.layers.Softmax(axis=-1)
                ])

    if flag == 1:
        model = inception_v3.InceptionV3(weights = 'imagenet', input_tensor = Input(shape = (IMG_WIDTH, IMG_HEIGHT, 3)))

    if flag == 2:
        model = ResNet50(weights = 'imagenet', input_tensor = Input(shape = (IMG_WIDTH, IMG_HEIGHT, 3)))

    if flag == 3:
        model = InceptionResNetV2(weights = 'imagenet', input_tensor = Input(shape = (IMG_WIDTH, IMG_HEIGHT, 3)))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode = 'min', 
                                            patience=5, verbose = 0, restore_best_weights=True)
    
    # Train model with early stopping 
    history = model.fit(x=np.array(X_train, np.float32), 
                        y=np.array(y_train, np.float32),validation_data=(np.array(X_val, np.float32), np.array(y_val, np.float32)),
                        epochs=n_epochs, callbacks= [callback])
    if flag == 0:
        filename = 'cnn.sav'
    if flag == 1:
        filename = 'inception_v3.sav'
    if flag == 2:
        filename = 'ResNet50.sav'
    if flag == 3:
        filename = 'InceptionResNetV2'

    os.makedirs(args.model_dir, exist_ok=True)
    pickle.dump(model, open(f"{args.model_dir}/{filename}", 'wb'))
    filename = f"{args.model_dir}/{filename}"
    return filename, history


def predict(filename,X_test,y_test):
    #load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    predictions = loaded_model.evaluate(x=np.array(X_test, np.float32), 
                                        y=np.array(y_test, np.float32))
    return predictions


def visualize(history):
    plt.plot(history.history['loss'], label = 'Train_loss')
    plt.plot(history.history['val_loss'], label = 'Val_loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()

    plt.plot(history.history['accuracy'], label = 'Train_acc')
    plt.plot(history.history['val_accuracy'], label = 'Val_acc')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.show()


def run_pipe():
    img_data, class_name = create_dataset(img_folder,IMG_WIDTH,IMG_HEIGHT)
    X_train, X_test, y_train, y_test, X_val, y_val = split(img_data, class_name)
    to_be_augmented = create_data_frame(X_train , y_train)
    X_train, y_train = data_augmentation(to_be_augmented)
    filename, history = train(flag, IMG_WIDTH, IMG_HEIGHT, X_train, y_train, X_val, y_val)
    visualize(history)
    _, accuracy = predict(filename, X_test, y_test)
    print("Test Accuracy: {accuracy}")

if __name__ == "__main__":
    run_pipe()