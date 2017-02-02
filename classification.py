#!/usr/bin/python

'''
Author: Iskandar Pashayev
Purpose: Train a neural network to classify images into two classes:
		- Contains the Consolidated Income Statement table
		- Does not contain the Consolidated Income Statement table
'''

import os
import numpy as np
import random as r
from PIL import Image
from optparse import OptionParser

# Helpers
from lib.utils import Data
from lib.classifier import Layers, NeuralNetwork
import lib.classifier as cl

'''
We only use 2 examples per PDF. Otherwise the false examples dominate the the true examples,
resulting in machine precision issues when calculating log probabilities of the True example.
'''
kNumExamples = 2
kNumClasses = 2 # only two classes: {is a table, is not a table}
kNumNeuronsHidden = 1

def parse_cmd_line():
    '''
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage=usage)
    parser.add_option('-n','--neural-network',
                      help='use a neural network to classify the data')
    parser.add_option('-s','--softmax',
                      help='use a softmax classifier to classify the data')
    '''
    return None

if __name__ == '__main__':
    args = parse_cmd_line();
    cwd_path = os.getcwd()
    with open('permnos.dat','r') as permnos:
        for line in permnos:
            datum = line.rstrip().split('\t') # read a line from the permmos list
            permno, name, year = datum[:3] # identifiers
            start, offset = [int(token) for token in datum[3:]] # values
            
            assert len(datum) == 5, 'Invalid data format for %s (permno: %s, year: %s, )' % (name,permno,year)

            # Load image in grayscale
            img = Image.open(permno + '/' + name + '.jpeg', 'r').convert('L')
            w, h = img.size # width, height

            # Get a two-dimensional numpy array of the pixel values of the image
            pixels = list(img.getdata())
            pixels = np.asarray(pixels)
            assert pixels.size % w == 0, 'Image imbalanced width-wise'

            # Set the data object we will train our neural network on
            dimensionality = offset*w # each class contains this number of pixels
            data = Data(kNumExamples, dimensionality, kNumClasses)

            '''
            Randomly choose 9 integers that are in [0,height-offset) \ (start-offset/2,start+offset/2).
            These 9 integers are starting locations for "False" examples, i.e. samples from the image
            that do not contain the financial table.

            We simulate this by randomly choosing from two integers which range to sample from:
            0 => sample from [0,start-offset/2)
            1 => sample from (start+offset/2,height-offset)

            Note to self: Double check the bounds for logical sense.
            '''
            false_starts = []
            range_one_upper = start - offset/2
            range_two_lower = start + offset/2
            range_two_upper = h - offset
            assert range_two_upper >= (range_two_lower + 9), 'Bad range when randomly choosing false examples'
            for i in xrange(kNumExamples-1):
                do_sample = True
                while do_sample:
                    choose_range = r.randint(0,1)
                    if choose_range == 0:
                        rand_int = r.randint(0, range_one_upper - 1)
                    else:
                        rand_int = r.randint(range_two_lower + 1, range_two_upper - 1)
                    if rand_int not in false_starts:
                        false_starts.append(rand_int)
                        do_sample = False

            X, y = np.zeros((data.N, data.D)), np.zeros(data.N, dtype='uint8')

            for i in xrange(kNumExamples-1):
                false_example_start = false_starts[i]
                print 'False example start at y:', false_example_start
                X[i] = pixels[false_example_start:false_example_start+dimensionality]
                y[i] = 0
            X[-1] = pixels[start:start+dimensionality]
            y[-1] = 1

            '''
            VERY IMPORTANT NOTE:
            We do not normalize here because there is so little variation in the data. Most of the
            We DO normalize the class scores for each example during training by setting the 
            large_input flag to True when calling classifyWithNeuralNetwork.
            '''
            data.set_data(X, y)
            data.preprocess()
            
            
            # A simple sanity check for types in the array
            '''
            if 0 in X: # Could cause divide by zero exceptions during preprocessing
                print 'X contains 0 values at'
                print np.argwhere(X == 0)
                print
            else:
                print 'No 0 values detected in the data'
            '''
            if np.isnan(X).any():
                print 'X contains NaN values at'
                print np.argwhere(np.isnan(X).any())
                print
            else:
                print 'No NaN values detected in the data'
            if np.isinf(X).any():
                print 'X contains Inf values at'
                print np.argwhere(np.isinf(X).any())
                print
            else:
                print 'No Inf values detected in the data'
                
            # Set up the neural network and classify the data
            # num_iterations = 1000
            # cl.classifyWithNeuralNetwork(data, kNumNeuronsHidden, num_iterations, True)

            # Classify the data with a softmax classifier
            cl.classifyWithSoftmax(data, large_input_bool=True)
