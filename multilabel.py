#
#   SED505 Design Pattern - Assignment 6
#   Multilabel - Simple Example
#   D.Audu, Y.Dong, K.Sethi
#   Due: 2023-12-10
#

import numpy as np 
import random 

class NeuralNetwork:
    
    numLabel = 3
    threshold = 0.68
    
    def predict(self, input):
        # simulating raw output of the NN
        logits = []
        # simulating NN hidden layers
        for i in range(0, self.numLabel):
            logits.append(((input[i] - input[i+1])) / 14)
        
        # last part of the output layer
        probs = self.output(logits)
        
        return probs
    
    def output(self, logits):
        # output layer
        probs = [self.activationFunction(x) for x in logits]
        return probs
    
    def activationFunction(self, x):
        # Sigmoid activation function 
        return 1/(1 + np.exp(-x))

def generateData(n):
    # generate n records of data
    random.seed(42)
    return [[random.randint(0, 100) for _ in range(5)] for i in range(0,n)]


nn = NeuralNetwork()

dataset = generateData(5)

for i, data in enumerate(dataset):
    probs = nn.predict(data)
    labels = [ 1 if prob > nn.threshold else 0 for prob in probs]
    print(f"Dataset #{i+1:2} prediction: {labels} with probabilities {[round(prob, 2) for prob in probs] }")