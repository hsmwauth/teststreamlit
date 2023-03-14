#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 11:21:55 2023

@author: auth
"""
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import helper_one as h

# creating artificial image
imagesize = 20
particlesize = 5

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


# blurring the image


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
axs[0,1].imshow(gauss, cmap='gray')
axs[1,0].imshow(laplacian2, cmap = 'gray')
axs[1,1].imshow(gausslaplace, cmap = 'gray')


fig.show()






