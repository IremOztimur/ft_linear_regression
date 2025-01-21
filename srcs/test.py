import tensorflow as tf
import pandas as pd
from preprocess import CustomStandardScaler, train_test_split

data = pd.read_csv('../data/data.csv')

scaler = CustomStandardScaler()
df = scaler.fit_transform(data)

X = df['km'].values.reshape(-1, 1)
y = df['price'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='sgd', loss='mse', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=500, verbose=0, validation_split=0.2)


loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Mean Absolute Error on test set: {mae:.2f}")

y_pred = model.predict(X_test)

y_test_original = y_test * scaler.std['price'] + scaler.mean['price']
y_pred_original = y_pred * scaler.std['price'] + scaler.mean['price']

for true, pred in zip(y_test_original[:5], y_pred_original[:5]):
    print(f"True Price: {true[0]:.2f}, Predicted Price: {pred[0]:.2f}")