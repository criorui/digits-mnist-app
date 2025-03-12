import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# Cargar el modelo
model = load_model("modelMNIST.h5")

def predict_image(image):
    # Preprocesar la imagen
    image = image.convert("L")  # Escala de grises
    image = image.resize((28, 28))  # Redimensionar
    image = np.array(image) / 255.0  # Normalizar
    image = image.reshape(1, 28, 28, 1)  # Añadir batch
    
    # Hacer la predicción
    predictions = model.predict(image)[0]
    predicted_class = np.argmax(predictions)
    confidence_scores = {str(i): round(pred * 100, 2) for i, pred in enumerate(predictions)}
    
    return predicted_class, confidence_scores

# Interfaz de Streamlit
st.title("Reconocimiento de Dígitos con CNN")
st.write("Sube un dibujo de un dígito (0-9) y el modelo intentará reconocerlo.")

uploaded_file = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)
    
    predicted_class, confidence_scores = predict_image(image)
    
    # Mostrar predicción en grande
    st.markdown(f"<h2 style='text-align: center; font-size: 50px;'>Predicción: {predicted_class}</h2>", unsafe_allow_html=True)
    
    # Mostrar gráfico de confianza
    fig, ax = plt.subplots()
    ax.bar(confidence_scores.keys(), confidence_scores.values(), color='blue')
    ax.set_xlabel("Clases")
    ax.set_ylabel("Confianza (%)")
    ax.set_title("Confianza en cada clase")
    st.pyplot(fig)
