import numpy as np
from abc import ABC, abstractmethod

class Loss(ABC):
    @abstractmethod
    def calculate(predictions, targets):
        pass
    
class MAE(Loss):
    def calculate(self, predictions, targets):
        return np.mean(np.abs(predictions - targets))
    def grad(self, predictions, targets):
        return np.where(predictions > targets, 1, -1) / len(predictions)