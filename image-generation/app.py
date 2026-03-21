import streamlit as st
from image_generator import generate_image

st.title("AI Image Agent")

prompt = st.text_input("Enter prompt")

if st.button("Generate Image"):

    path = generate_image(prompt)

    st.image(path)