import streamlit as st
from ai_agent import run_agent

st.title("AI Image and text Agent")
prompt = st.text_input("Enter prompt")

if st.button("Generate"):
    response = run_agent(prompt)
    st.write(response["text"])

    if response["final_prompt"]:
        st.caption(f"Final image prompt: {response['final_prompt']}")

    if response["image_path"]:
        st.image(response["image_path"], caption="Generated image", use_container_width=True)
        st.code(response["image_path"])
    
