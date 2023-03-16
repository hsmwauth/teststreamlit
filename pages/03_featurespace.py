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
import numpy as np

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
size = st.sidebar.slider('Markersize',0,20,2)

# creating the plot
fig, ax = plt.subplots(figsize=(5.5, 5.5))
plt.xlabel('size')
plt.ylabel('sharpness')
ax.set_xscale('log')
ax.set_yscale('log')



ax.scatter(images_notreceived[0],images_notreceived[1], c = images_notreceived[2], alpha=opacity, s=size, cmap='gray',edgecolors='k',linewidth=0.5) #beachte reihenfolge, da die ersten übermahlt werden.
ax.scatter(images_received[0], images_received[1], c= images_received[2], alpha=opacity, s=size, cmap='gray',edgecolors='g',linewidth=0.5)
legend = ['not received images', 'received images']

groundtruth = st.sidebar.checkbox("Show groundtruth", False)
if groundtruth:
    # load database
    db_gt = pd.read_feather('./data/groundtruth.feather')
    size = db_gt['size'].to_numpy()
    sharp = db_gt['laplace_var'].to_numpy()
    ax.scatter(size,sharp,  c ='y', alpha=0.2, s=2)
    name = 'groundtruth'
    if name in legend:
        pass
    else:
        legend.append(name)


hyperplane = st.sidebar.checkbox("Show hyperplane", False)
if hyperplane:
    x2=np.linspace(2,10,100)
    m = -2.4/4
    b = 0.1
    x1 = np.e**x2
    fx = np.e**(m*x2 + b)
    ax.plot(x1,fx, linewidth=0.2)
    name = 'hyperplane'
    if name in legend:
        pass
    else:
        legend.append(name)

#[x1, x2] = h.getinterestingness(db)

#origin = [6.0,-6.0]
#ax.quiver(origin[0], origin[1], c.weight_size, c.weight_sharpness, scale=10)

#ax.plot(x1,x2)

ax.legend(legend, loc='lower left')
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
cropped_image = np.asarray(cropped_image)/255 # normalizing dtype=uint8 to 1

# CALCULATE FEATURE -> sharpness
    # gaussian filter
stencil_gauss = (1/16)*np.array(([1,2,1],[2,4,2],[1,2,1]))
gauss = ndimage.convolve(cropped_image, stencil_gauss, mode='wrap')

    # laplace
stencil_laplace = np.array([[1,1,1],[1,-8,1],[1,1,1]])
gausslaplace = ndimage.convolve(gauss,stencil_laplace, mode='wrap')
laplace_variance = np.var(gausslaplace)
''')

st.subheader('Combining sharpness and size')
st.info('Sharpness und Size werden mit einer Linearkombination der Einzelfeatures vereinigt.')
st.code('''
        # Calculate the interestingness in log-space
feature1 = np.log(size_pixelcount)
feature2 = np.log(laplace_variance)
p = np.array([feature1,feature2])
distance = np.cross(c.p2-c.p1,p-c.p1)/np.linalg.norm(c.p2-c.p1)
interestingness = distance


def ordering(data):

    # ORDER DATAFRAME DUE TO INTERESTINGNESS in ascendeing order
    data.sort_values(by=['interestingness'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    # print(data.tail())

    # CREATE INTEGER ORDER-VECTOR
    n_rows = len(data.filename)
    order = np.linspace(start=1,
                        stop=n_rows,
                        num=n_rows,
                        dtype=int)
    data['order'] = order  # add the integer-order-vector to the dataframe
    print('Created and stored following table:')
    print(data.describe())
    #plt.hist(data.sharpness,bins=100)
    # plt.show()
    return data
        ''')
