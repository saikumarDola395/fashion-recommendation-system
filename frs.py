from zipfile import ZipFile
import os
import glob
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D ,Dense
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from tensorflow.keras import layers, Sequential

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D


from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

from keras.models import load_model
savedModel=load_model(r'D:\finalweights.h5')

import pickle

with open('names.pkl','rb') as f: all_image_names = pickle.load(f)
with open('features.pkl','rb') as f: all_features = pickle.load(f)

import glob
# Correctly generate the list of image file paths
image_directory = r'D:\images'
image_paths_list = glob.glob(os.path.join(image_directory, '*'))



def preprocess_image(img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def extract_features(model, preprocessed_img):
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features


def recommend_fashion(input_image_path, all_features, all_image_names, model, top_n=5):
    preprocessed_img = preprocess_image(input_image_path)
    input_features = extract_features(model, preprocessed_img)
    print(input_features,type(input_features))
    k=[]

    similarities = [1 - cosine(input_features, feature) for feature in all_features]
    top_indices = np.argsort(similarities)[-top_n-1:]

    # Ensure the subplot grid is dynamically sized to accommodate all images
    num_images = top_n + 1  # Number of images to display includes the input image and top N recommendations
    

    # Adjust the loop to correctly handle subplot positions
    rec_position = 2  # Start placing recommendation images from the second position
    for idx in top_indices:
        if all_image_names[idx] == os.path.basename(input_image_path):
            continue  # Skip the input image

        k.append(image_paths_list[idx])
        rec_position += 1

        # Stop if we've filled up the intended number of recommendations
        if rec_position > top_n + 1:
            break
        
    return k

    
# input_image_path = r"C:\Users\saiku\OneDrive\Desktop\testing_images\28806.jpg"
# # Adjust to an actual file path
# k = recommend_fashion(input_image_path, all_features, all_image_names, savedModel, top_n=4)

