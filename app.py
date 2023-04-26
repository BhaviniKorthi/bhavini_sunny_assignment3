import streamlit as st
from PIL import Image
import os


st.header("Latent Space Visualisation")
st.sidebar.markdown("Here's some additional information about the visualisation:")



files = os.listdir("plots")
num_plots = len(files)
image_dict = {}
for i in range(num_plots):
    index = (i+1)*5
    image_dict[index] = Image.open(f"plots/plot{i+1}.png")
selected_index = st.slider("Epochs", 5, num_plots*5, 5, step=5)
st.image(image_dict[selected_index], caption=f"Plot {selected_index//5}", use_column_width=True)