#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 09:27:10 2023

@author: auth
"""

import pandas as pd
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import helper_one as h

th_value=167

# import all images from decisionmaking
db = pd.read_feather(r'./data/decisionmaking.feather')

size = list()
laplace_var = list()
min_laplace = list()
max_laplace = list()
gauss_var = list()
min_grad = list()
max_grad = list()
colorcode = list()

# loop over database
for row in db.itertuples():
    
    # klasse
    klasse = row.klasse
    if klasse == 'Good':
        color = 'green'
    elif klasse == 'Bad':
        color = 'red'
    colorcode.append(color)
    
    #load image
    imagepath = r'./data/train_particles/' + row.cropedimage
    image = Image.open(imagepath)
    image = np.asarray(image)/255 # normalizing dtype=uint8 to 1
    
    # gaussian filter
    stencil_gauss = (1/16)*np.array(([1,2,1],[2,4,2],[1,2,1]))
    gauss = ndimage.convolve(image, stencil_gauss, mode='wrap')
    gauss_abs = np.absolute(gauss)
    
    min_gauss = np.min(gauss_abs)
    min_grad.append(min_gauss)
    
    max_gauss = np.max(gauss_abs)
    max_grad.append(max_gauss)
    
    gauss_variance = np.var(gauss_abs)
    gauss_var.append(gauss_variance)
    
    # laplace
    stencil_laplace = np.array([[1,1,1],[1,-8,1],[1,1,1]])
    gausslaplace = ndimage.convolve(gauss,stencil_laplace, mode='wrap')
    laplace_variance = np.var(gausslaplace)
    laplace_var.append(laplace_variance)
    
    max_lap = np.max(gausslaplace)
    max_laplace.append(max_lap)
    
    min_lap = np.max(gausslaplace)
    min_laplace.append(min_lap)
    
    # getting size
    binary_image = np.where(image > (th_value/255), 1, 0)
    n_zeros = np.count_nonzero(binary_image == 0) # here is a particle in the pixel
    size.append(n_zeros)
    
db['size'] = size
db['laplace_var'] = laplace_var
db['min_laplace'] = min_laplace
db['max_laplace'] = max_laplace
db['gauss_var'] = gauss_var
db['min_grad'] = min_grad
db['max_grad'] = max_grad
db['colorcode'] = colorcode

x = db['size'].to_numpy()

#y = db['gauss_var'].to_numpy()
#y = db['min_grad'].to_numpy()
#y = db['max_grad'].to_numpy()
y = db['laplace_var'].to_numpy()
#y = db['max_laplace'].to_numpy()
#y = db['min_laplace'].to_numpy()
colorcode = db['colorcode'].to_numpy()


# plotte alles (inter-active-plot)
fig = h.iap(x,y,colorcode,db)
#plt.savefig('plot.pdf')

db.to_feather('groundtruth.feather')