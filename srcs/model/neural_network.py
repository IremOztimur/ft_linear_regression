from model.layer import Dense
from model.loss import MAE
from model.optimizer import SGD
import numpy as np

class NeuralNetwork:
    layers: Dense
    
    def __init__(self, loss_function=MAE()):
        self.layers = []
        self.loss_function = loss_function
        self.optimizer = SGD(learning_rate=0.01)
        self.history = {'loss': [], 'val_loss': []}
        
    def add(self, layer):
        if not isinstance(layer, Dense):
            raise TypeError("The added layer must be an instance of "
                            f"class Dense. Found: {layer}")
        self.layers.append(layer)
        
    def train(self, X, y, epochs, batch_size=0, validation_data=None, patience=3):
        if batch_size == 0:
            batch_size = X.shape[0]
        
        if validation_data:
            X_val, y_val = validation_data
            best_loss = np.inf
            patience_counter = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                predictions = self.layers[0].forward(X_batch)
                loss = self.loss_function.calculate(predictions, y_batch)
                epoch_loss += loss
                gradients = self.loss_function.grad(predictions, y_batch)
                self.layers[0].backward(gradients)
                self.optimizer.update_params(self.layers[0])
                
            epoch_loss /= (X.shape[0] // batch_size)
            self.history['loss'].append(epoch_loss)
            
            if validation_data:
                y_pred = self.predict(X_val)
                val_loss = self.loss_function.calculate(y_val, y_pred)
                self.history['val_loss'].append(val_loss)
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter == patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        return self.history
    def predict(self, X):
        return self.layers[0].forward(X)