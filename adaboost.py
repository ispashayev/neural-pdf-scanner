#!/usr/bin/python

'''
Author: Iskandar Pashayev
Purpose: Learn the weights for the labels to identify the company labels
'''

import os
import numpy as np
import integrate as i

if __name__ == '__main__':
    cwd_path = os.getcwd()
    with open('permnos.dat','r') as permnos:
        for line in permnos:
            datum = line.rstrip().split('\t')
            permno, name = datum[:2]
            assert (len(datum)-2) % 4 == 0, "Invalid box labels for permno: %s" % permno
            num_boxes = (len(datum) - 2) / 4
            pixels = i.integrate_img(permno, name)
            rows, cols = pixels.shape
            examples = np.matrix(np.zeros(pixels.shape))
            for b in num_boxes:
                start_row, start_col = datum[2+4*b:2+4*b+1]
                end_row = start_row + datum[4+4*b]
                end_col = start_col + datum[5+4*b]
                examples[start_row:end_row+1,start_col:end_col+1] = 1
