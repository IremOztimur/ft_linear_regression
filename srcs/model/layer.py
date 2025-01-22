import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self, n_inputs, n_neurons):
        self.theta1 = np.zeros((n_inputs, n_neurons))
        self.theta0 = np.zeros((1, n_neurons))
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
    
    @abstractmethod
    def forward(self, inputs):
        pass
    
        
class Dense(Layer):
    def __init__(self, n_inputs, n_neurons):
        super().__init__(n_inputs, n_neurons)
        
    def forward(self, inputs):
        self.inputs = inputs
        self.linear_output = np.dot(inputs, self.theta1) + self.theta0
        
    def backward(self, gradients):
        self.dtheta1 = np.dot(self.inputs.T, gradients)
        self.dtheta0 = np.sum(gradients, axis=0, keepdims=True)
        self.dinputs = np.dot(gradients, self.theta1.T)