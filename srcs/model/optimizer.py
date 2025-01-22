import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def update_params(self, layer):
        pass

class SGD(Optimizer):
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def update_params(self,layer):
        layer.theta1 += -self.learning_rate * layer.dtheta1
        layer.theta0 += -self.learning_rate * layer.dtheta0