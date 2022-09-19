
# Corn Maize Leaf Disease Prediction

Project ini dibuat untuk memprediksi suatu gejala pada daun tanaman jagung. Ada 4 kelas yaitu, Healthy, Gray Leaf Spot, Common Rust, dan Blight. Project ini menggunakan teknik CNN (Convulutional Neural Network) yang baik untuk memprediksi sebuah gambar.




## Demo


Demo Aplikasi: https://agungyogasetiawan-cor-corn-maize-leaf-disease-prediction-5hv55d.streamlitapp.com/

Code Model Building CNN: [Klik ini untuk Code pembuatan model CNN di Google Colab](https://colab.research.google.com/drive/16vroV5lLsZIMYTduFHiYOXs9uKrw7zGc)
## Code

Import library yang diperlukan

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

Membuat nilai konstanta

```bash
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
```

Memasukkan folder data dengan fungsi image_dataset_from_directory fungsinya untuk memasukkan folder dataset yang berisi gambar dari local
```bash
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    './Data Resize',
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    seed=123,
    shuffle=True
)
```

Method class_names untuk menampilkan class yang ada di folder dataset

```bash
class_names = dataset.class_names
#['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
```

Membuat fungsi load model 
```bash
def load_model():
  model=tf.keras.models.load_model('model_corn_maize.h5')
  return model
model=load_model()
```

Fungsi dari streamlit yaitu untuk upload sebuah file
```bash
file = st.file_uploader("Please upload an file image", type=["jpg", "png", "jpeg"])
```

Fungsi untuk memprediksi sebuah gambar, dan gambar tersebut dirubah menjadi sebuah array untuk dapat dibaca oleh mesin. Fungsi ini mengembalikan prediksi kelas dan skor prediksi
```bash
def import_and_predict(image, model):
    
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    score = round(100 * (np.max(predictions[0])), 2)
          
    return predicted_class, score
```

Sebuah perintah kondisi, jika file tidak kosong maka akan menampilkan gambar yang kita upload dan memanggil fungsi prediksi yang mengembalikan prediksi kelas dan skor prediksi nya. Apbila kosong maka hanya menampilkan sebuah teks
```bash
if file is not None:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predicted_class, score = import_and_predict(image, model)
    st.write(f'This image most likely belongs to {predicted_class} with a {score} % confidence.')
else:
    st.text("Please upload an image file")
```
    
