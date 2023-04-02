import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
header = st.container()
image = st.container()
with header:
    st.title('Image segmentation')
    st.text('Input the lung image and then it will return the segmentation of that image')
with image:
    st.title('input image')
    im = st.file_uploader('image', type = ['pnj', 'jpg', 'txt'])
    if im is not None:
        im = Image.open(im)
        img = np.array(im)
        st.image(im)