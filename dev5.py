#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:09:04 2023

@author: auth
"""

'''
input is the image counter and the database,
from that it creates a fake image width all detected particles
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os.path
from skimage import io


import constants as c

number= '000004'

#Getting DB
db = pd.read_feather(c.DBPATH)

# Getting Imagename
imagename = 'Os7-S1 Camera'+ number

# Filter the database according to the particles occuring in the image
db_img = db[db['filename'] == imagename]

print(db_img)

# creating empty dummy image
dummy_image = np.ones((c.IMAGESIZE_X, c.IMAGESIZE_Y), dtype=np.uint8)*c.BACKGROUNDCOLOR

# loop over every entry and add it to the image
db_img = db_img.reset_index()  # make sure indexes pair with number of rows

for index, row in db_img.iterrows():
    # Get local Variables
    left = int(row['left'])
    right = int(row['right'])
    top = int(row['top'])
    bottom = int(row['bottom'])
    # size_pixelcount = row['size_pixelcount']
    # sharpness = row['sharpness']
    # interestingness = row['interestingness']
    order = row['order']
    print(order)
    
    width = right - left
    height = bottom - top
    
    # st.write([left, right, top, bottom])
    
    # cropped image in case of existance
    cropped_imgpath = c.CROPPEDIMAGEPATH + '/' + str(order) + '_' + imagename + '.png'
    
    # whole image path in case of existance
    imagepath = c.CROPPEDIMAGEPATH + '/' + imagename + '.png'
    
    #offset = np.array((left, top))
    
    if os.path.exists(imagepath): # wenn das ganz Image schon runtergeladen wurde wird natürlich dieses dargestellt
        img = io.imread(imagepath)
    elif os.path.exists(cropped_imgpath): # wenn das gecroppte image existiert
        cropped_img = io.imread(cropped_imgpath)
        dummy_image[top : (top + height), left : (left + width)] = cropped_img


    else:
        cropped_img = np.ones((height, width), dtype=np.uint8)
        dummy_image[top : (top + height), left : (left + width)] = cropped_img
    
dummy_image = Image.fromarray(dummy_image, 'L')


draw = ImageDraw.Draw(dummy_image)
for index, row in db_img.iterrows():
    left = int(row['left'])
    right = int(row['right'])
    top = int(row['top'])
    bottom = int(row['bottom'])
    # size_pixelcount = row['size_pixelcount']
    # sharpness = row['sharpness']
    # interestingness = row['interestingness']
    order = row['order']

    text = str(order)
    textwidth, textheight = draw.textsize(text)
    marginx = marginy = 2
    x = right - textwidth -marginx
    y = bottom - textheight - marginy
    draw.text((x, y), text)
     
plt.imshow(dummy_image,cmap='gray')