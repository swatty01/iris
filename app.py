import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model


model = load_model('iris_model.keras')
with open('scaler.pkl', 'rb') as f:
  scaler = pickle.load(f)
with open('encoder.pkl', 'rb') as f:
  encoder = pickle.load(f)

st.title("🌸 Iris Flower Species Predictor")
st.write("Enter flower measurement to predict the species")

sepal_length = st.number_input("Sepal Length (cm)", 4.0, 8.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", 2.0, 4.5, step=0.1)
petal_length = st.number_input("Petal Length (cm)", 1.0, 7.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", 0.1, 2.5, step=0.1)

if st.button("Predic"):
  input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
  input_scaled = scaler.transform(input_data)

  prediction = model.predict(input_scaled)
  pred_class_index = np.argmax(prediction)
  pred_label = encoder.inverse_transform([[1 if i == pred_class_index  else 0 for i in range(3)]])[0][0]

  st.success(f"🌻 Predicted Species: **{pred_label}**")
