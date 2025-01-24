# Linear Regression Car Price Prediction

This project is part of the 42 curriculum. The aim of this project is to introduce you to the basic concept behind machine learning. For this project, you will create a program that predicts the price of a car by using a linear function trained with a gradient descent algorithm.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Predicting Car Prices](#predicting-car-prices)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this project is to implement a linear regression model to predict car prices based on their mileage. The model is trained using a gradient descent algorithm. This project will help you understand the fundamentals of machine learning, data preprocessing, and model evaluation.

## Project Structure

```
.
├── data
│   └── data.csv
├── srcs
│   ├── model
│   │   ├── layer.py
│   │   ├── loss.py
│   │   ├── metric.py
│   │   ├── neural_network.py
|   |   ├── preprocess.py
│   │   └── regression.py
│   ├── predict.py
│   ├── train.py
│   └── Makefile
└── README.md
```

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/IremOztimur/ft_linear_regression.git
    cd ft_linear_regression/srcs
    ```

2. Create a virtual environment and install the required packages:
    ```sh
    make virtual
    ```

or only install the required packages:
    ```
    make install
    ```

3. Create a directory to save the model
    ```
    make create
    ```
## Usage

### Training the Model

To train the model, run the `train.py` script with the required arguments:

```sh
python train.py --data ../data/data.csv --epochs 150 --learning_rate 0.1 --model_name car_price_model
```

### Predicting Car Prices

To predict car prices, run the `predict.py` script with the path to the saved model:

```sh
python predict.py --model depo/car_price_model.npy
```

You will be prompted to enter the mileage value. The program will predict the car price based on the entered mileage. You can continue to enter mileage values or type 'quit' to exit the program.



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.