#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 12:04:39 2018

@author: victor
"""

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('letras/a0.jpg', cv.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')

img = img.astype('float32') / 255

a = np.ones((25,25))

m_x, m_y = np.random.randint(0, 25, 2) # coordenadas da média e o desvio
desvio = np.random.randint(12, 25)

for lin in range(a.shape[0]):
    for col in range(a.shape[1]):
        a[lin][col] = np.exp(-((lin-m_x)**2/(2*desvio**2)+(col-m_y)**2/(2*desvio**2)))
        #print('a[{}][{}]: {}'.format(lin, col, a[lin][col]))
plt.imshow(a, cmap='gray'), plt.show()


img1 = np.multiply(img, a)
plt.imshow(img1, cmap='gray'), plt.show()