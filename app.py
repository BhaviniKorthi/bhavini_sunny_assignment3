import streamlit as st
from PIL import Image
import os


st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page", ["Slider", "Reconstructed Images"])
st.sidebar.markdown("   ")
files = os.listdir("plots")
num_plots = len(files)
image_dict = {}
for i in range(num_plots):
    index = (i+1)*5
    image_dict[index] = Image.open(f"plots/plot{i+1}.png")

if page == "Slider":
    st.title("Latent Space Visualisation")
    for i in range(num_plots):
        index = (i+1)*5
        image_dict[index] = Image.open(f"plots/plot{i+1}.png")
    selected_index = st.slider("Epochs", 5, num_plots*5, 5, step=5)
    st.image(image_dict[selected_index], caption=f"Plot {selected_index//5}", use_column_width=True)
else:
    st.title("Reconstructed Images")
    st.image(Image.open("Reconstructed/Figure_1.png"), caption="PlotX", use_column_width=True, width=300)


