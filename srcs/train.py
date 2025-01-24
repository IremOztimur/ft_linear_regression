from model.layer import Dense
from model.optimizer import SGD
from preprocess import CustomStandardScaler, train_test_split
from model.regression import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pyfiglet


def main():
    parser = argparse.ArgumentParser(description="Training Program")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset (CSV format).")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate for training.")
    parser.add_argument("--model_name", type=str, required=True, help="The name of the model")
    args = parser.parse_args()
    
    data = pd.read_csv(args.data)

    scaler = CustomStandardScaler()
    df = scaler.fit_transform(data)

    X = df['km'].values.reshape(-1, 1)
    y = df['price'].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    model = LinearRegression()
    model.optimizer.learning_rate = args.learning_rate
    model.mean = scaler.mean
    model.std = scaler.std
    
    word = pyfiglet.figlet_format("Linear Regression Training")
    print(f"\033[95m{word}\033[00m")
    print("*" * 40)
    print("\033[95mTraining the model...\033[00m")
    print("*" * 40)
    history = model.train(X_train, y_train, epochs=args.epochs, validation_data=(X_test, y_test), patience=3)
    print("\033[92mTraining completed.\033[00m")
    print("*" * 40)
    
    print("Training Loss: ", history['loss'][-1])
    print("Validation Loss: ", history['val_loss'][-1])
    print("*" * 40)
    print("\n")
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.close
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    y_test_original = y_test * scaler.std['price'] + scaler.mean['price']
    y_pred_test_original = y_pred_test * scaler.std['price'] + scaler.mean['price']
    y_train_original = y_train * scaler.std['price'] + scaler.mean['price']
    y_pred_train_original = y_pred_train * scaler.std['price'] + scaler.mean['price']

    
    print("\n")
    for true, pred in zip(y_test_original[:5], y_pred_test_original[:5]):
        print(f"True Price: {true[0]:.2f}, Predicted Price: {pred[0]:.2f}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train_original, color='green', label='Actual data (Train)')
    plt.plot(X_train, y_pred_train_original, color='orange', label='Predicted line (Train)')
    plt.xlabel('km')
    plt.ylabel('price')
    plt.legend()
    plt.show()
    
    model.save_model(f"depo/{args.model_name}.npy")
    
    
if __name__ == '__main__':
    main()