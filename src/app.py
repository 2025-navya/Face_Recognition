import streamlit as st
from predict_face import predict_face
import os
from PIL import Image

st.set_page_config(page_title="Face Recognition", layout="centered")
st.title("👤 Face Recognition Demo")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    img.save("temp.jpg")  # Save temporarily for predict_face

    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting..."):
        result = predict_face("temp.jpg")

    st.subheader("🔍 Prediction Result")
    st.success(f"**{result['label']}** ({result['confidence'] * 100:.2f}% confidence)")

    st.subheader("📊 Class Probabilities")
    st.bar_chart(result["all_predictions"])
