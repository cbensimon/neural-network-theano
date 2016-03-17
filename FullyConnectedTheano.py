# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:15:32 2016

@author: 3000120
"""

import numpy as np
from theano import *
import theano.tensor as T

class Layer:
    
    def __init__(self, n_in, n_out, sigma):
        self.w = shared(np.random.randn(n_out,n_in))
        self.b = shared(np.random.randn(n_out, 1))
        self.sigma = sigma
        
    def out(self, ins):
        return self.sigma(self.w.dot(ins) + self.b)
        
class Network:
    
    def __sigmoid(x):
        return 1.0/(1+T.exp(-x))    
    
    def __init__(self, shapes, sigma=self.sigmoid, eps=0.1):
        self.layers = [Layer(shapes[i], shapes[i+1], sigma) for i in range(len(shapes)-1)]
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        self.out = self.x
        for layer in self.layers:
            self.out = layer.out(self.out)
        
    def __cost(self, ins, labels):
        return -T.mean(T.log(self.out[zip(range(len(labels)), self.y)]))
        
    def fit(datax, datay, batch_size=10, epochs=10):
        nb_train = int(np.ceil(datax.shape[0]/float(batch_size)))
        
                
        
        for epoch in range(epochs):
            for _ in range(nb_train):
                
        
        
def sigmoid(x):
    return 1.0/(1+T.exp(-x))       
        
nn = Network([256,30,10], sigmoid)

ins = shared(np.random.randn(256,1))

out = nn.out(ins)
