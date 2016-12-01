'''
Author: Iskandar Pashayev
Purpose: For each pixel value in an image, calculate the sum of all the pixel values
above and to the left, including the value in the current place. This allows the feature
detector used in the Viola Jones algorithm to be scale invariant.
'''

import os
import numpy as np
from PIL import Image

'''
Returns a 3-tuple of numpy matrix objects
'''
def integrate_img(permno, name):

    img = Image.open(name + '.jpeg', 'r')
    w, h = img.size # width, height
    pixels = list(img.getdata())
    red, green, blue = zip(*pixels)
    red = np.matrix([ red[i*w:i*w+w] for i in range(h) ])
    green = np.matrix([ green[i*w:i*w+w] for i in range(h) ])
    blue = np.matrix([ blue[i*w:i*w+w] for i in range(h) ])

    for i in range(1,h):
        red[i,:] = red[i-1,:] + red[i,:]
        green[i,:] = green[i-1,:] + green[i,:]
        blue[i,:] = blue[i-1,:] + blue[i,:]

    for i in range(1,w):
        red[:,i] = red[:,i-1] + red[:,i]
        green[:,i] = green[:,i-1] + green[:,i]
        blue[:,i] = blue[:,i-1] + blue[:,i]

    return (red,green,blue)

if __name__ == '__main__':
    integrate_img(10000, 'ACF INDS INC')