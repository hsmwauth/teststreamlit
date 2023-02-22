#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 12:49:19 2023
@author: auth

Following questions should be answered on this page:
    * How many cropped images are received.
    * How many cropped images are we expecting to receive until certain time.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np

import helper_one as h
import constants as c

# Set title and Info for this slide
st.title('Featurespace')
st.info('Displaying the featurespace on base what the AID detected. Received particles get displayed in green. Not received one in black. You can change some Diagramsettings in the sidebar. The black arrow is the representation of the gradient of the Orderfunktion.')

# import database
db = pd.read_feather(c.DBPATH)

# datapreparation for displaying the featurspace
[images_received, images_notreceived] = h.getfeaturespace(db)

# displaying properties
opacity = st.sidebar.slider('Opacity', 0.0, 1.0, 0.5)
size = st.sidebar.slider('Markersize',0,10,2)

# creating the plot
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.scatter(images_notreceived[0],images_notreceived[1], c = 'k', alpha=opacity, s=size) #beachte reihenfolge, da die ersten übermahlt werden.
ax.scatter(images_received[0], images_received[1], c= 'g', alpha=opacity, s=size)
plt.legend(['not received images', 'received images'])
plt.xlabel('size')
plt.ylabel('sharpness')

#[x1, x2] = h.getinterestingness(db)

#origin = [6.0,-6.0]
#ax.quiver(origin[0], origin[1], c.weight_size, c.weight_sharpness, scale=10)

#ax.plot(x1,x2)

# displaying the plot
st.pyplot(fig)

#--------------------------------------------------------------------
st.header('Calculation Details')

# Display information on the calculation of size
st.subheader('Feature Calculation Size')
st.info('Für die Berechnung der Größe wurde auf Basis des Flatfields (Berechnung des Thresholdes über den ersten 5 Bildern des Experiments) die Pixel, die diesen Threshold überschreiten aufsummiert. Mit der fokalen Tiefe kann die Size direkt in eine Fläche/Größe übersetzt werden.')
st.code('''def calculate_th():
    path = c.path2th_images
    imagenames = c.th_images

    list_of_mean = []

    for x in imagenames:
        if exists(path + x):
            img = mpimg.imread(path +x) #imports as array of float32 (0...1)
            mean = np.mean(img)
            list_of_mean.append(mean)
            
        else:
            print('File existiert nicht')
    
    if list_of_mean == []:
        th = c.threshold
    else:
        th = np.mean(list_of_mean)
    return th''')

st.code('''
# CALCULATE FEATURE -> size_pixelcount
binary_image = np.where(cropped_image > threshold, 1, 0)
# n_ones= np.count_nonzero((binary_image) == 1) # here is no particle in the pixel
n_zeros = np.count_nonzero(binary_image == 0) # here is a particle in the pixel
size_pixelcount = n_zeros
''')

# Display information on calculation of sharpness
st.subheader('Feature Calculation Sharpness')
st.info('Für die Berechnung der Schärfe wurde die Variance des Laplace Filters vom Gauss Filter vewendet.')

st.code('''
# CALCULATE FEATURE -> sharpness
start = time.process_time()
blur = cv2.GaussianBlur(cropped_image, (5, 5), 0)  # apply Gauss-Filter
blur_norm = (blur-np.min(blur))/(np.max(blur)-np.min(blur)) #normalize beween 0,1# normalizing between 0,1
laplacian = cv2.Laplacian(blur_norm, cv2.CV_64F)  # apply Laplace-Filter
laplacian_norm = (laplacian-np.min(laplacian))/np.max(laplacian)-np.min(laplacian) # normalizing between 0,1
sharpness = np.log10(laplacian_norm.var()) # TODO aply divie by zero and apply the log10
''')

st.subheader('Combining sharpness and size')
st.info('Sharpness und Size werden mit einer Linearkombination der Einzelfeatures vereinigt.')
st.code('''
weight_size = 0.8  # [0...1]
weight_sharpness = 0.5  # [0...1]
        
def calc_interestingness(feature_sharpness, feature_size):
    interestingness = np.multiply(c.weight_sharpness * feature_sharpness, c.weight_size * feature_size)
    return interestingness
        ''')
