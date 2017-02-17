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

# Helpers
from lib.parser import Parser
from lib.utils import Data
from lib.classifier import Layers, NeuralNetwork
import lib.classifier as cl

'''
We only use 2 examples per PDF. Otherwise the false examples dominate the the true examples,
resulting in machine precision issues when calculating log probabilities of the True example.
'''
kNumClasses = 2 # only two classes: {is a table, is not a table}
kNumNeuronsHidden = 1


'''
Randomly choose 9 integers that are in [0,height-offset) \ (start-offset/2,start+offset/2).
These 9 integers are starting locations for "False" examples, i.e. samples from the image
that do not contain the financial table.

We simulate this by randomly choosing from two integers which range to sample from:
0 => sample from [0,start-offset/2)
1 => sample from (start+offset/2,height-offset)

Note: The last argument is unnecessary, but included for readability.
Note to self: Double check the bounds for logical sense.
'''
def set_examples(data, examples, start, offset, img_width, num_false):
    X, y = np.zeros((data.N, data.D)), np.zeros(data.N, dtype='uint8')
    skipped_examples = []
    for i in xrange(len(examples)):
        example_pix = examples[i]
        range_one_upper = start - offset/2
        range_two_lower = start + offset/2
        range_two_upper = len(example_pix) / img_width - offset
        try:
            # Check that it's possible to generate enough unique false examples
            assert (range_one_upper - offset) + (range_two_upper - range_two_lower - offset) >= num_false
            'Bad range when randomly choosing false examples - skipped.'
        except AssertionError as ae:
            skipped_examples.append(i)
            print ae.args[0]
            continue

        # Generate locations for random false examples. Allows for some overlap but not duplicates.
        false_starts = []
        for j in range(len(num_false)):
            while True:
                choose_range = r.randint(0,1)
                if choose_range == 0:
                    rand_int = r.randint(0, range_one_upper - 1)
                else:
                    rand_int = r.randint(range_two_lower + 1, range_two_upper)
                if rand_int not in false_starts:
                    false_starts.append(rand_int)
                    break

        # Set false examples (i.e. boxes without net income table)
        for j in range(len(false_starts)):
            print 'False example start at y:', false_starts[j]
            X[i+j] = example_pix[false_start:false_start + data.D]
            y[i+j] = 0

        # Set true example (i.e. box for net income table)
        X[i+num_false+1] = example_pix[start:start + data.D]
        y[i+num_false+1] = 1

    # Remove skipped examples and print count for number skipped
    np.delete(X, skipped_examples, axis=0)
    np.delete(y, skipped_examples, axis=0)
    
    '''
    VERY IMPORTANT NOTE:
    We do not normalize the features here because there is so little variation in the data.
    We DO normalize the class scores for each example during training by setting the
    large_input flag to True when calling classifyWithNeuralNetwork.
    '''
    data.set_data(X, y)

'''
Opens a one-dimensional array of pixels for each image in permnos and combines them on
separate rows for each image. Returns a data object that contains a numpy array of the combined
pixels.
'''
def load_data(path_to_permnos, num_false):
    examples = []
    with open(path_to_permnos,'r') as permnos:
        for line in permnos:
	    datum = line.rstrip().split('\t') # read line from the permmos data file
            permno, name, year = datum[:3] # identifiers
	    start, offset = [ int(token) for token in datum[3:] ] # true example labels
	                
	    # Load image in grayscale
	    img = Image.open('data/' + permno + '/' + name + '.jpeg', 'r').convert('L')
            w, h = img.size # width, height
            try:
                assert w == uniform_w and offset == uniform_offset,
                permno + ' ' + year + ' not uniform to other examples - skipped.'
            except NameError: uniform_w, uniform_offset = w, offset
            except AssertionError as ae: print ae.args[0]; continue
            
            # Get a list of the pixel values of the image
	    curr_pixels = list(img.getdata())
            try:
                # have no idea why it was like this.. should be following line
                # assert curr_pixels.size % w == 0,
                assert len(curr_pixels) % w == 0,
                permno + ' ' + year + ' imbalanced width-wise - skipped.'
            except AssertionError as ae: print ae.args[0]; continue
            
	    examples.append((curr_pixels, start))
    
    # Set the data object we will train our neural network on
    dimensionality = uniform_offset * uniform_w # each example contains this number of pixels
    data = Data(len(examples)*(num_false + 1), dimensionality, kNumClasses)
    set_examples(data, examples, uniform_offset, uniform_w, num_false)
    
    return data


if __name__ == '__main__':
    parser = Parser()
    parser.get_args() # parse command line arguments for config file, output file, & verbose flag
    parser.read() # parse config file

    data = load_data(parser.get_data_path(), parser.get_num_false())
    data.preprocess() # performs mean subtraction on the data
    
    # A simple sanity check for types in the array
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
    num_iterations = parser.get_num_iter()
    hidden_layers = parser.get_hidden()
    cl.classifyWithNeuralNetwork(data, hidden_layers, num_iterations, True)
    
