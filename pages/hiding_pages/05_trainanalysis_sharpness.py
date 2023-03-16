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
from scipy import ndimage
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
image = Image.open(path2file)


# gaussian filter
stencil_gauss = (1/16)*np.array(([1,2,1],[2,4,2],[1,2,1]))
gauss = ndimage.convolve(image, stencil_gauss, mode='wrap')

# laplace filter
stencil_laplace1 = np.array([[0, 1, 0],[1, -4, 1], [0, 1, 0]]) #
stencil_laplace2 = np.array([[1,1,1],[1,-8,1],[1,1,1]]) # additionally for diagonal corners
laplacian1 = ndimage.convolve(image, stencil_laplace1, mode = 'wrap')
laplacian2 = ndimage.convolve(image, stencil_laplace2, mode = 'wrap')

gausslaplace = ndimage.convolve(gauss,stencil_laplace2, mode='wrap')

fig, axs = plt.subplots(2,2)
axs[0,0].imshow(image, cmap = 'gray')
axs[0,0].set_title('original')
axs[0,0].set_xticks([], [])

axs[0,1].imshow(gauss, cmap='gray')
axs[0,1].set_title('gauss(origininal)')
axs[0,1].set_xticks([], [])

axs[1,0].imshow(laplacian2, cmap = 'gray')
axs[1,0].set_title('laplace(origninal)')
axs[1,0].set_xticks([], [])

axs[1,1].imshow(gausslaplace, cmap = 'gray')
axs[1,1].set_title('laplace(gauss(original))')
axs[1,1].set_xticks([], [])

st.pyplot(fig)

st.write(str(int(np.var(gauss))))
