import numpy as np

'''
Defines a Data object used for classification tasks where
	N = number of points per class
	D = dimensionality of data
	K = number of classes
'''
class Data(object):
    def __init__(self, N=100, D=2, K=3):
        self.N = N
        self.D = D
        self.K = K
        self.X = None
        self.y = None
        self.preprocessed = True
    def set_data(self, X, y):
        self.X = X
        self.y = y
    def construct_toy_data(self):
        self.X = np.zeros((self.N*self.K, self.D)) # data matrix (each row = single example)
        self.y = np.zeros(self.N*self.K, dtype='uint8') # class labels
        for j in xrange(self.K):
            ix =  range(self.N*j, self.N*(j+1))
            r = np.linspace(0.0, 1, self.N) # radius
            t = np.linspace(j*4, (j+1)*4, self.N) + np.random.randn(self.N)*0.2 # theta
            self.X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
            self.y[ix] = j
    # Left unimplemented for now
    def preprocess(self, mean_subtraction=True, normalization=False, pca=False, whitening=False):
        assert self.X is not None and self.y is not None, 'Uninitialized data'
        if mean_subtraction:
            self.X -= np.mean(self.X, axis=0)
        if normalization:
            self.X /= np.std(self.X, axis=0)
        if pca:
            ''' Unimplemented '''
            pass
        if whitening:
            ''' Unimplemented '''
            pass


'''
Defines a Layers object used to keep track of different layer attributes within a neural network
'''
class Layers(object):
    def __init__(self, D, K):
        self.K = K
        self.num_layers = 1
        self.layers = [(D,K)]
    '''
    Add a layer of n neurons to the network right before the output layer
    '''
    def add_layer(self, n):
        self.layers[-1:] = [(self.layers[-1][0],n), (n,self.K)]
        self.num_layers += 1
