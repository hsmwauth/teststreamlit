#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:25:01 2023
Hier sollen die Trianingsimages nochmals unterschucht werden
@author: auth
"""

import streamlit as st
import pandas as pd
import os
import glob
from PIL import Image
import numpy as np

st.title('Featureanalysis Size')


# importing database
db = pd.read_feather(r'./data/decisionmaking.feather')


# displaying dataframe
#st.dataframe(db)

# get filenames of the croped particles
files = os.listdir('./data/train_particles')

# slider to pick one particle
img_crop = st.select_slider('',files)

# slider threshold
th = st.slider('',  min_value= 0, max_value= 255, value=167)

path2file = r'./data/train_particles/' + img_crop

st.write(img_crop)

# load the image
img = Image.open(path2file)
st.image(img)

# CALCULATE FEATURE -> size_pixelcount
image = np.asarray(img)/255 # normalizing dtype=uint8 to 1
binary_image = np.where(image > (th/255), 1, 0)
# n_ones= np.count_nonzero((binary_image) == 1) # here is no particle in the pixel
n_zeros = np.count_nonzero(binary_image == 0) # here is a particle in the pixel
size_pixelcount = n_zeros

image2display = Image.fromarray((binary_image*255).astype('uint8'), 'L')
st.image(image2display)


