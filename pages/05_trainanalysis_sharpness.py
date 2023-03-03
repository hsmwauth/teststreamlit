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
st.info('Hier sollen das Kriterium Schärfe nochmals erörtert werden.')
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
st.subheader('Original Particle')
st.image(img)

# Normalizing
norm_img = np.asarray(img)/255 # normalizing dtype=uint8 to 1

st.subheader('Blurring and Normalization')
kernelsize = st.slider('Kernel size',1,15,5)
blur = cv2.blur(norm_img, (kernelsize, kernelsize))
blur_img = h.normalizing(blur)
st.image(blur_img)

# Calculate gradient (L1 - norm)
st.subheader('Gradient-based operators')
gradients = np.gradient(blur)
grad = abs(np.sqrt(gradients[0]**2 + gradients[1]**2))
grad_img = h.normalizing(grad)
st.image(grad_img)

# making the Histogram on the gradient
start, end = st.select_slider('uint8 Range for gradient', options = np.linspace(0,256,257).tolist(), value=(0,256))
histogram = cv2.calcHist([grad_img],[0],mask=None,histSize=[int(end-start)], ranges=[int(start), int(end)])
st.bar_chart(histogram, use_container_width=True)

st.subheader('Laplace-based operators')

laplace = cv2.Laplacian(blur,cv2.CV_64F)
laplace_img = h.normalizing(laplace)

# making the Histogram on the laplace
start_l, end_l = st.select_slider('uint8 Range for laplace', options = np.linspace(0,256,257).tolist(), value=(0,256))
histogram = cv2.calcHist([laplace_img],[0],mask=None,histSize=[int(end-start)], ranges=[int(start_l), int(end_l)])
st.bar_chart(histogram, use_container_width=True)

st.image(laplace_img)
