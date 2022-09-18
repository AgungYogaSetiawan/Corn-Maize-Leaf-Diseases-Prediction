# -*- coding: utf-8 -*-
"""Corn Maize Leaf Disease Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16vroV5lLsZIMYTduFHiYOXs9uKrw7zGc

### Import Required Dependencies
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from PIL import Image
from tensorflow.keras import models, layers
from keras.models import load_model

IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'Data Resize',
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    seed=123,
    shuffle=True
)

class_names = dataset.class_names

"""### Build Web App Using Streamlit"""

import streamlit as st

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('model_corn_maize.h5')
  return model
model=load_model()

st.write("""
# Corn Maize Leaf Diseases Classification
"""
)

file = st.file_uploader("Please upload an file image", type=["jpg", "png", "jpeg"])
st.set_option('deprecation.showfileUploaderEncoding', False)

def import_and_predict(image, model):
    
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    score = round(100 * (np.max(predictions[0])), 2)
          
    return predicted_class, score

if file is not None:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predicted_class, score = import_and_predict(image, model)
    st.write(f'This image most likely belongs to {predicted_class} with a {score} % confidence.')
else:
    st.text("Please upload an image file")

