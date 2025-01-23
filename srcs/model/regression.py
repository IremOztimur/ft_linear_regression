from .neural_network import NeuralNetwork
from .layer import Dense
from .loss import MAE


class LinearRegression(NeuralNetwork):
    def __init__(self, loss_function=MAE()):
        super().__init__(loss_function)
        self.add(Dense(1, 1))
        