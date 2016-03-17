# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:15:32 2016

@author: 3000120
"""

import numpy as np
from theano import *
import theano.tensor as T
from theano.tensor.nnet import sigmoid
from arftools import *

class Layer:
    
    cpt = 0
    
    def __init__(self, n_in, n_out, sigma):
        self.w = shared(np.random.randn(n_in, n_out).astype(np.float32), name='w'+str(Layer.cpt))
        self.b = shared(np.random.randn(n_out,).astype(np.float32), name='b'+str(Layer.cpt))
        Layer.cpt += 1
        self.sigma = sigma
        self.params = [self.w, self.b]
        
    def out(self, ins):
        return self.sigma(ins.dot(self.w) + self.b)
        
class Network: 
   
    def __init__(self, shapes, sigma=sigmoid):
        self.layers = [Layer(shapes[i], shapes[i+1], sigma) for i in range(len(shapes)-1)]
        self.x = T.matrix('x')
        self.y = T.imatrix('y')
        self.out = self.x
        for layer in self.layers:
            self.out = layer.out(self.out)
        self.params = [param for layer in self.layers for param in layer.params]
       
    def __cost(self, ):
        return T.mean(T.sum((self.out - self.y)**2, axis=1))
       
    def fit(self, datax, datay, batch_size=10, epochs=100, eps=1):
        nb_train = int(np.ceil(datax.shape[0]/float(batch_size)))
       
        grad = T.grad(self.__cost(), self.params)
        fGrad = function([self.x, self.y], grad)
        fOut = function([self.x], self.out)
        fCost = function([self.x, self.y], self.__cost())
       
        for epoch in range(epochs):
            print 'epoch'
            for _ in range(nb_train):
                batch_indexes = np.random.choice(range(len(datax)), batch_size, replace=False)
                xBatch = datax[batch_indexes]
                yBatch = datay[batch_indexes]
                gradValue = fGrad(xBatch, yBatch)
               
                for i in range(len(self.params)):
                    param = self.params[i]
                    param.set_value((param.get_value() - gradValue[i]*eps).astype(np.float32))
                    
            outValue = fOut(datax)
            yPred = np.argmax(outValue, axis=1)
            yReal = np.argmax(datay, axis=1)
            print 'score: ' + str((yPred == yReal).mean())
            print 'cost: ' + str(fCost(datax, datay))
            print ''
                  
    def predict(self, datax):
        fOut = function([self.x], self.out)
        
        outValue = fOut(datax)
        
        return outValue
               
                  
       
nn = Network([256,30,10], sigmoid)
datax, datay = load_usps()

X = datax.astype(np.float32)

Y = np.zeros((datay.shape[0], 10)).astype(np.int32)
for i in range(datay.shape[0]):
    Y[i, datay[i]] = 1

gradValue = nn.fit(X, Y)
outValue = nn.predict(X)
yPred = np.argmax(outValue, axis=1)
