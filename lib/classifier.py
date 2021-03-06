'''
Author: Iskandar Pashayev
Purpose: Build gGeneralized classifiers for a data object

Resources (partial code + theory) used from cs231n.github.io by Andrej Karpathy
'''

import numpy as np
import matplotlib.pyplot as plt
from utils import Data, Layers

'''
Debugging runtime warnings:
1) overflow
2) divide by zero
3) invalid values
'''
import sys
import warnings
warnings.filterwarnings('error')

'''
Parent class for different types of classifiers.
Subclasses implement train() and evaluate() methods.
'''
class Classifier(object):
    def __init__(self, data, num_iterations):
        ''' Data '''
        self.X = data.X
        self.y = data.y
        self.N = data.N
        self.num_examples = self.X.shape[0]
        print self.X.shape
        ''' Hyperparameters '''
        self.step_size = 1e-0
        self.reg = 1e-3 # regularization strength
        self.num_iterations = num_iterations

class NeuralNetwork(Classifier):
    '''
    We use a layers object to get the number of neurons per layer
    '''
    def __init__(self, data, layers, num_iterations=10000):
        super(NeuralNetwork, self).__init__(data, num_iterations)
        ''' Parameters '''
        self.num_layers = layers.num_layers
        self.W_list, self.b_list = [], []
        for i in xrange(self.num_layers):
            self.W_list.append(0.01 * np.random.randn(layers.layers[i][0], layers.layers[i][1]))
            self.b_list.append(np.zeros((1, layers.layers[i][1])))
    
    def train(self, large_scores=False):
        assert len(self.W_list) == len(self.b_list) # simple sanity check
        '''
        Forward pass: Evaluate class scores, e.g. W_2 * max(0,W_1*X+b), where
        max is an elementwise operation and
        0 is an [N x K] matrix of zeros.
        '''
        for iteration in xrange(self.num_iterations):
            if iteration % 10 == 0: print 'Beginning iteration:', iteration
            layers = [self.X] # input layer
            for i in xrange(self.num_layers-1):
                W_i, b_i, layer_i = self.W_list[i], self.b_list[i], layers[i]
                layers.append(np.maximum(0, np.dot(layer_i, W_i) + b_i)) # ReLU activation
            scores = np.dot(layers[-1], self.W_list[-1]) + self.b_list[-1] # output layer
            if large_scores: # then needs normalizing to prevent overflow during exponentiation
                scores -= np.max(scores, axis=1, keepdims=True)
                
            # Compute the loss: average cross-entropy and regularization
            ''' Can probably move this part into parent class '''
            exp_scores = np.exp(scores) # element-wise exponentiation
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
            try:
                correct_logprobs = -np.log(probs[range(self.num_examples),self.y])
            except RuntimeWarning as rw:
                print 'runtime warning on iteration', iteration
                print 'class probabilities:'
                print probs
                sys.exit(rw)
            data_loss = np.sum(correct_logprobs) / self.num_examples
            reg_loss = sum([0.5*self.reg*np.sum(W*W) for W in self.W_list])
            loss = data_loss + reg_loss
            if iteration % 100 == 0: print "loss: %f" % loss

                
            # Compute derivative of loss wrt scores
            dscores = probs
            dscores[range(self.num_examples),self.y] -= 1
            dscores /= self.num_examples
            
            # Backpropagate dscores to dparameters
            dW_list, db_list, dhiddens = [], [], []
            for i in xrange(self.num_layers-1,0,-1):
                W_i, b_i, hidden_i = self.W_list[i], self.b_list[i], layers[i]
                
                '''
                Finding the gradients of the terms of W_i*max(0,hidden_i) + b_i in
                the loss function. (i.e. wrt W_i, b_i, and hidden_i

                Note: this code hasn't been tested with 3 layer networks (yet)
                '''
                try:
                    dlayer = dhiddens[0]
                except:
                    dlayer = dscores
                dW_i = np.dot(hidden_i.T, dlayer) # gradient of loss wrt W_i
                db_i = np.sum(dlayer, axis=0, keepdims=True) # gradient of loss wrt b_i
                dhidden = np.dot(dlayer, W_i.T) # gradient of loss wrt hidden layer
                dhidden[hidden_i <= 0] = 0 # backprop the ReLU non-linearity
                
                dhiddens.insert(0, dhidden) # insertions might be fucked up
                dW_list.insert(0, dW_i)
                db_list.insert(0, db_i)
                
            dW_0 = np.dot(self.X.T, dhiddens[0]) # self.X.T should be equal to layers[0]
            db_0 = np.sum(dhiddens[0], axis=0, keepdims=True)
            dW_list.insert(0, dW_0)
            db_list.insert(0, db_0)
                    
            for i in xrange(self.num_layers):
                dW_list[i] += self.reg*self.W_list[i] # Add regularization gradient distribution
                self.W_list[i] += -self.step_size * dW_list[i]
                self.b_list[i] += -self.step_size * db_list[i]
            
    def evaluate(self):
        inp = self.X
        for i in xrange(self.num_layers-1):
            inp = np.maximum(0, np.dot(inp, self.W_list[i]) + self.b_list[i])
        scores = np.dot(inp, self.W_list[-1]) + self.b_list[-1]
        predicted_class = np.argmax(scores, axis=1)
        return 'training accuracy: %.2f' % (np.mean(predicted_class == self.y))


class Softmax(Classifier):
    def __init__(self, data, num_iterations=200):
        super(Softmax, self).__init__(data, num_iterations)
        ''' Parameters '''
        self.W = 0.01 * np.random.randn(data.D, data.K)
        self.b = np.zeros((1, data.K))
    
    def train(self, large_scores=False):
        for i in xrange(self.num_iterations):
            # Compute the class scores for a linear classifier
            scores = np.dot(self.X, self.W) + self.b
            if large_scores: # needs normalizing
                scores -= np.max(scores, axis=1, keepdims=True)

            # Compute the loss: average cross-entropy loss and regularization
            exp_scores = np.exp(scores) # unnormalized probabilities
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # normalized
            correct_logprobs = -np.log(probs[range(self.num_examples),self.y])
            data_loss = np.sum(correct_logprobs) / self.num_examples
            reg_loss = 0.5*self.reg*np.sum(self.W*self.W)
            loss = data_loss + reg_loss
            if i % 10 == 0: print "iteration %d: loss %f" % (i, loss)

            # Computing the Analytic Gradient with Backpropagation
            dscores = probs
            dscores[range(self.num_examples), self.y] -= 1
            dscores /= self.num_examples
            dW = np.dot(self.X.T, dscores)
            db = np.sum(dscores, axis=0, keepdims=True)
            dW += self.reg*self.W

            # Performing a parameter update
            self.W += -self.step_size * dW
            self.b += -self.step_size * db

    def evaluate(self):
        scores = np.dot(self.X, self.W) + self.b
        predicted_class = np.argmax(scores, axis=1)
        result_str = 'training accuracy: %.2f' % np.mean(predicted_class == self.y)
        return result_str


def classifyWithSoftmax(data, num_iterations=0, large_input_bool=False):
    # Training a Softmax Classifier to classify the data
    if num_iterations > 0: softmax = Softmax(data, num_iterations)
    else: softmax = Softmax(data)
    softmax.train(large_input_bool)
    accuracy = softmax.evaluate()
    print accuracy

def classifyWithNeuralNetwork(data, num_neurons, num_iterations=0, large_input_bool=False):
    # Training a Neural Network to classify the data
    layers = Layers(data.D, data.K)
    if len(num_neurons) == 0: num_neurons = 10 # default to a single hidden layer with 10 neurons
    for hidden_set in num_neurons:
        layers.add_layer(num_neurons)
    if num_iterations > 0: neural_network = NeuralNetwork(data, layers, num_iterations)
    else: neural_network = NeuralNetwork(data, layers)
    neural_network.train(large_input_bool)
    accuracy = neural_network.evaluate()
    print accuracy
    
if __name__ == '__main__':
    toy_2d_data = Data()
    toy_2d_data.construct_toy_data()
    toy_2d_data.preprocess() # No actual need for preprocessing here.
    
    '''
    # Visualizing the data
    X, y = toy_2d_data.X, toy_2d_data.y
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()
    '''

    # classifyWithSoftmax(toy_2d_data)
    classifyWithNeuralNetwork(toy_2d_data, num_neurons=100)
