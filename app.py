import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Cargar el modelo
model = load_model("modelMNIST.h5")

# Función para preprocesar y predecir
def predict_image(image):
    image = ImageOps.grayscale(image)  # Convertir a escala de grises
    image = image.resize((28, 28))  # Redimensionar
    image = np.array(image) / 255.0  # Normalizar
    image = image.reshape(1, 28, 28, 1)  # Ajustar dimensiones
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class

# Interfaz con Streamlit
st.title("Clasificador de Dígitos - MNIST")

uploaded_file = st.file_uploader("Sube una imagen de un número escrito a mano", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    if st.button("Predecir"):
        prediction = predict_image(image)
        st.write(f"**Número predicho:** {prediction}")
