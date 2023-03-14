#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 08:29:25 2023

@author: auth
"""

import streamlit as st
import numpy as np
import pandas as pd
from skimage import io
import cv2

import constants as c
import helper_one as h

st.title('Order by Sharpness')
st.info('Got ***Good-Particles*** ordered by their interestingness from AP am 04.03.2023.')

# import csv
db = pd.read_csv(c.PATHORDERED)
minimum = int(db['order'].min())
maximum = int(db['order'].max())
init = 10

order = st.slider('Pic an order:', minimum, maximum, 10)
st.write(order)

img = db[db['order'] == order]['img_name'].iloc[0]
imgpath = r'./data/train_particles' + '/' + img
st.write(imgpath)
image = io.imread(imgpath)
st.image(image)

# Normalizing
norm_img = np.asarray(image)/255 # normalizing dtype=uint8 to 1

st.subheader('Blurring and Normalization')
kernelsize = st.slider('Kernel size',1,15,5)
blur = cv2.blur(norm_img, (kernelsize, kernelsize))
blur_img = h.normalizing(blur)
st.image(blur_img)