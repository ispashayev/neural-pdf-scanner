import numpy as np
import matplotlib.pyplot as plt

'''
N = number of points per class
D = dimensionality of data
K = number of classes
'''
class Data(object):
    def __init__(self, N=100, D=2, K=3):
        self.N = N
        self.D = D
        self.K = K
    def getDim(self):
        return self.D
    def getNumClasses(self):
        return self.K
    '''
    # Need to implement this
    def preprocess(self):
        pass
    '''
    def construct_toy_data(self):
        self.X = np.zeros((self.N*self.K, self.D)) # data matrix (each row = single example)
        self.y = np.zeros(self.N*self.K, dtype='uint8') # class labels
        for j in xrange(self.K):
            ix =  range(self.N*j, self.N*(j+1))
            r = np.linspace(0.0, 1, self.N) # radius
            t = np.linspace(j*4, (j+1)*4, self.N) + np.random.randn(self.N)*0.2 # theta
            self.X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
            self.y[ix] = j
    
class Classifier(object):
    def __init__(self, data):
        self.data = data
        ''' Hyperparameters '''
        self.step_size = 1e-0
        self.reg = 1e-3
        
    '''
    We implement the following methods in the subclasses of Classifier
    train()
    evaluate()
    '''
    
class Softmax(Classifier):
    def __init__(self, data):
        super(Softmax, self).__init__(data)
        ''' Parameters '''
        self.W = 0.01 * np.random.randn(data.getDim(), data.getNumClasses())
        self.b = np.zeros((1, data.getNumClasses()))
    
    def train(self):
        X, y = self.data.X, self.data.y
        num_examples = X.shape[0]
        for i in xrange(200):
            # Compute the class scores for a linear classifier
            self.scores = np.dot(X, self.W) + self.b

            # Compute the loss: average cross-entropy loss and regularization
            exp_scores = np.exp(self.scores) # unnormalized probabilities
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # normalized
            correct_logprobs = -np.log(probs[range(num_examples),y])
            data_loss = np.sum(correct_logprobs) / num_examples
            reg_loss = 0.5*self.reg*np.sum(self.W*self.W)
            loss = data_loss + reg_loss
            if i % 10 == 0:
                print "iteration %d: loss %f" % (i, loss)

            # Computing the Analytic Gradient with Backpropagation
            dscores = probs
            dscores[range(num_examples), y] -= 1
            dscores /= num_examples
            dW = np.dot(X.T, dscores)
            db = np.sum(dscores, axis=0, keepdims=True)
            dW += self.reg*self.W

            # Performing a parameter update
            self.W += -self.step_size * dW
            self.b += -self.step_size * db
            

    def evaluate(self):
        scores = np.dot(self.data.X, self.W) + self.b
        predicted_class = np.argmax(self.scores, axis=1)
        result_str = 'training accuracy: %.2f' % np.mean(predicted_class == y)
        return result_str

class Layers(object):
    def __init__(self, D, K):
        self.K = K
        self.num_layers = 1
        self.layers = [(D,K)]
    def add_layer(self, num_neurons):
        self.layers[-1:] = [(self.layers[-1][0], num_neurons),(num_neurons,self.K)]
        self.num_layers += 1
    
class NeuralNetwork(Classifier):
    def __init__(self, data, layers):
        super(NeuralNetwork, self).__init__(data)
        ''' Parameters '''
        self.W_list, self.b_list, self.hidden = [], [], []
        for i in xrange(layers.num_layers):
            self.W_list.append(0.01 * np.random.randn(layers[i])) # problem with unpacking tuple here maybe
            self.b_list.append(np.zeros((1, layers[i][1])))
    
    def train(self):
        X, y = self.data.X, self.data.y
        # Evaluate class scores (forward pass)
        for i in xrange(layers.num_layers-1):
            ''' NOTE: fix the following crime against humanity '''
            self.hidden.append(np.maximum(0, np.dot(X, self.W_list[i]) + self.b_list[i])) # (Stacking) ReLU activation
        self.scores = np.dot(self.hidden[-1], self.W_list[-1]) + self.b_list[-1]

        # Compute the loss: average cross-entropy and regularization
        ''' Can probably move this part into parent class '''
        exp_scores = np.exp(self.scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_logprobs = -np.log(probs[range(num_examples),y])
        data_loss = np.sum(correct_logprobs)/num_examples

        ''' PAUSED HERE '''
        reg_loss = 0.5*self.reg*np.sum(
        
        
    def evaluate(self):
        pass

def classifyWithSoftmax(data):
    # Training a Softmax Classifier to classify the data
    softmax = Softmax(data)
    softmax.train()
    accuracy = softmax.evaluate()
    print accuracy

def classifyWithNeuralNetwork(data):
    layers = Layers(data.getDim(), data.getNumClasses())
    layers.add_layer(100)
    neural_network = NeuralNetwork(data, layers)
    
if __name__ == '__main__':
    toy_2d_data = Data()
    toy_2d_data.construct_toy_data()
    X, y = toy_2d_data.X, toy_2d_data.y

    # data.preprocess()
    '''
    # Visualizing the data
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()
    '''

    #classifyWithSoftmax(toy_2d_data)
    classifyWithNeuralNetwork(toy_2d_data)
