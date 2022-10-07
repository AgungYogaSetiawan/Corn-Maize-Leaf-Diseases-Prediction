
# Corn Maize Leaf Disease Prediction

This project was created to predict a symptom in the leaves of corn plants. There are 4 classes, namely, Healthy, Gray Leaf Spot, Common Rust, and Blight. This project uses a good CNN (Convulational Neural Network) technique to predict an image.




## Demo


Application Demo: https://agungyogasetiawan-cor-corn-maize-leaf-disease-prediction-5hv55d.streamlitapp.com/

Code Model Building CNN: [Click here for CNN modeling code on Google Colab](https://colab.research.google.com/drive/16vroV5lLsZIMYTduFHiYOXs9uKrw7zGc)

To run the app in your local machine, first go to your directory and write this syntax in terminal
```bash
streamlit run yournamefile.py
```

## Code

Import required libraries

```bash
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import streamlit as st
from PIL import Image
from tensorflow.keras import models, layers
from keras.models import load_model
```

Creating a constant value

```bash
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
```

Entering data folder with function image_dataset_from_directory function to enter dataset folder containing images from local
```bash
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    './Data Resize',
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    seed=123,
    shuffle=True
)
```

The class_names method to display the classes in the dataset folder

```bash
class_names = dataset.class_names
#['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
```

Creating a load model function
```bash
def load_model():
  model=tf.keras.models.load_model('model_corn_maize.h5')
  return model
model=load_model()
```

The function of streamlit is to upload a file
```bash
file = st.file_uploader("Please upload an file image", type=["jpg", "png", "jpeg"])
```

The function is to predict an image, and the image is converted into an array to be read by the machine. This function returns class prediction and predicted score
```bash
def import_and_predict(image, model):
    
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    score = round(100 * (np.max(predictions[0])), 2)
          
    return predicted_class, score
```

A condition command, if the file is not empty it will display the uploaded image and call the prediction function which returns the class prediction and its prediction score. If it is empty then it only displays a text
```bash
if file is not None:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predicted_class, score = import_and_predict(image, model)
    st.write(f'This image most likely belongs to {predicted_class} with a {score} % confidence.')
else:
    st.text("Please upload an image file")
```
    
