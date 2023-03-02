#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 08:11:13 2023

@author: auth
"""

import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

import helper_one as h

st.title('Featureanalysis Sharpness')

st.markdown('''
            
            * Gradient-based operators
            
            * Laplacian-based operators
            
            * Wavelet-based
            
            * statistic-based operators
            
            * Discrete cosine transform based operators
            
            * Miscellaneous
            ''')
            
# importing database
db = pd.read_feather(r'./data/decisionmaking.feather')

files = db['cropedimage']


# slider files
particle = st.select_slider('',  files)

path2file = r'./data/train_particles/' + particle

st.write(path2file)

# load the image
img = Image.open(path2file)

# Display the image
st.image(img)

st.subheader('Gradient-based operators')

# Normalizing
norm_img = np.asarray(img)/255 # normalizing dtype=uint8 to 1

# Calculate gradient (L1 - norm)
gradients = np.gradient(norm_img)
grad = abs(np.sqrt(gradients[0]**2 + gradients[1]**2))
grad_img = h.normalizing(grad)

# displaying the gradient image
st.image(grad_img)

# making the Histogram on the gradient
start, end = st.select_slider('uint8 Range', options = np.linspace(0,256,257).tolist(), value=(0,256))
histogram = cv2.calcHist([grad_img],[0],mask=None,histSize=[int(end-start)], ranges=[int(start), int(end)])
st.bar_chart(histogram, use_container_width=True)

