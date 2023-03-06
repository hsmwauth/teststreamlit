#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 13:27:38 2023

@author: auth
"""

import numpy as np
import streamlit as st

# creating artificial image
imagesize = st.slider('Choosing a image size in pixels',1,100,50)
particlesize = st.slider('Choose a particle size in pixels',1,imagesize,10)

particlevalue = 255
backroundvalue = 140

particle = np.ones((particlesize,particlesize))*particlevalue
backround = np.ones((imagesize,imagesize))*backroundvalue

shape_p = particle.shape
shape_i = backround.shape

left = int((shape_i[0]/2)-(shape_p[0]/2))
right = left + shape_p[0]
top = int((shape_i[1]/2) - (shape_p[1]/2))
bottom = top + shape_p[1]

backround[left:right,top:bottom] = particle

image = backround.astype(dtype='uint8')

st.image(image)


# blurring the image


# gradient analysis


# 