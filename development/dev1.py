#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 08:31:57 2023

@author: auth
"""
import numpy as np
import matplotlib.pyplot as plt


x1 = np.linspace(0,10000,100)

# Logistic Fuction:
L = 1 # supremum of the values of the function
k = 1/1000 # logistic growth rate or steepness of the curve
x0 = 200 #  the x value of the function's midpoint
f = L/(1+np.exp((-1*k)*(x1-x0)))

# Plot the logistic function
fig, ax = plt.subplots(1,1)
ax.plot(x1,f)
ax.set_xscale('log')
#ax.set_yscale('log')
ax.set_xlim(0,10000)

fig.show()