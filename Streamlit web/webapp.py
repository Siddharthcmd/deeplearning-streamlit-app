import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Bean Image Classifier")
st.text("Provide URL of bean Image for image classification")

@st.cache(allow_output_mutation=True) #Adding the model to the cache to save time.
def load_model():
    
    """This function will retrive the model from the system and will load it in model""" 
    
    model = tf.keras.models.load_model('C:/Users/test/models/') #You can put your own location here.
    return model

with st.spinner('Loading Model Into Memory....'):
    model = load_model()

classes=['angular_leaf_spot','bean_rust','healthy']

def decode_img(image):
    
    """This function will reshape the image."""

    img = tf.image.decode_jpeg(image, channels=3)  
    img = tf.image.resize(img,[224,224])
    return np.expand_dims(img, axis=0)

path = st.text_input('Enter Image URL to Classify.. ')
if path is not None:
    content = requests.get(path).content

    st.write("Predicted Class :")
    with st.spinner('classifying.....'):
        label =np.argmax(model.predict(decode_img(content)),axis=1)
        st.write(classes[label[0]])    
    st.write("")
    image = Image.open(BytesIO(content))
    st.image(image, caption='Classifying Bean Image', use_column_width=True)