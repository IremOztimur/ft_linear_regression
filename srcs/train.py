from model.layer import Dense
from model.optimizer import SGD
from model.neural_network import NeuralNetwork
from preprocess import CustomStandardScaler, train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    data = pd.read_csv('../data/data.csv')

    scaler = CustomStandardScaler()
    df = scaler.fit_transform(data)

    X = df['km'].values.reshape(-1, 1)
    y = df['price'].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)
    
    nn = NeuralNetwork()
    nn.add(Dense(1, 1))
    
    print("Training the model...")
    history = nn.train(X_train, y_train, epochs=200, validation_data=(X_test, y_test), patience=3)
    print("Training completed.")
    
    print("Training Loss: ", history['loss'][-1])
    print("Validation Loss: ", history['val_loss'][-1])
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    
if __name__ == '__main__':
    main()