import tensorflow as tf
from tensorflow.keras.models import Squential
from tensorflow.keras.layers import Dense
import pandas
import numpy as np

data = {
    "price": [100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000],
    "bedrooms": [1, 2, 3, 4],
    "bathrooms": [1, 2, 3],
    "sqft": [1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
    "location": ["city", "suburb", "rural"]
}
df = pandas.DataFrame(data)
df = pandas.get_dummies(df, columns=["location"], drop_first=True)

X = df.drop("price", axis=1).values
y = df["price"].values

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

model = tf.keras.Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(X, y, epochs=50, batch_size=4, verbose=1)

model.save("house_price_predictor.h5")