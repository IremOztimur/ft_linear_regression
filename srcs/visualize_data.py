import pandas as pd
import matplotlib.pyplot as plt
from preprocess import CustomStandardScaler

df = pd.read_csv('../data/data.csv')

print(df.head())
print("**************************************\n")
print(df.info())
print("**************************************\n")
print(df.describe())

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(df['km'], df['price'], 'o', label='data')
plt.xlabel('km')
plt.ylabel('price')
# plt.show()

print("**************************************\n")
scaler = CustomStandardScaler()
normalized_df = scaler.fit_transform(df)

plt.figure(figsize=(10, 6))
plt.plot(normalized_df['km'], normalized_df['price'], 'o', label='data')
plt.xlabel('km')
plt.ylabel('price')
plt.show()
