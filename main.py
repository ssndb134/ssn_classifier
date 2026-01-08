import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("./model.keras")
def predict_image(f):
    i1 = Image.open(f).resize((224,224)).convert("RGB")
    i1a = tf.keras.preprocessing.image.img_to_array(i1)
    i1a = i1a / 255.0
    i1a = np.expand_dims(i1a, axis=0)
    pred = model.predict(i1a)
    return "dog" if pred > 0.5 else "cat"

st.title("SSN Classifier")

f = st.file_uploader("Upload image")

if st.button("Classify"):
    r = predict_image(f)
    st.image(f, width=400)
    st.header(f"# **This is image of a :rainbow[{r}]**")
