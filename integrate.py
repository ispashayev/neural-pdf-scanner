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

    img = Image.open(permno + '/' + name + '.jpeg', 'r').convert('L')
    w, h = img.size # width, height
    pixels = list(img.getdata())
    pixels = np.matrix([ pixels[i*w:i*w+w] for i in range(h) ])
    for i in range(1,h):
        pixels[i,:] = pixels[i-1,:] + pixels[i,:]

    for i in range(1,w):
        pixels[:,i] = pixels[:,i-1] + pixels[:,i]

    return pixels


if __name__ == '__main__':
    print integrate_img(10000, 'ACF INDS INC')
