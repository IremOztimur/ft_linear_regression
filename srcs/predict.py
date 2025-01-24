import argparse
import pyfiglet
from model.regression import LinearRegression
from preprocess import CustomStandardScaler
from model.layer import Dense
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Prediction Program")
    parser.add_argument("--model", type=str, required=True, help="Path to the saved model (npy format).")
    args = parser.parse_args()
    scaler = CustomStandardScaler()
    
    lg = LinearRegression()
    model = lg.load_model(args.model)
    scaler.mean = model.mean
    scaler.std = model.std

    word = pyfiglet.figlet_format("Linear Regression")
    print(f"\033[95m{word}\033[00m")
    mileage = float(input("Enter the mileage value: "))

    
    X_scaled = scaler.transform(np.array([[mileage]]))

    y_pred = model.predict(X_scaled)
    y_pred_org = y_pred * model.std['price'] + model.mean['price']

    print(f"\033[95mPredicted Price: {y_pred_org[0][0]:.2f}\033[00m")
    
    

if __name__ == "__main__":
    main()