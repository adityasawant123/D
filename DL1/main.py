import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import LeakyReLU, Dropout

# Load the dataset
file_path = "Boston_Housing.csv"
data = pd.read_csv(file_path)

# Preprocess the data
X = data.drop(columns=['LSTAT'])  # Features
y = data['LSTAT']  # Target



# Preprocess the data again
X = data.drop(columns=['MEDV'])  # Features
y = data['MEDV']  # Target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the model
model = Sequential([
    Dense(128, activation=LeakyReLU(alpha=0.1), input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),  # Adjust dropout rate
    Dense(64, activation=LeakyReLU(alpha=0.1)),
    Dropout(0.2),  # Adjust dropout rate
    Dense(32, activation=LeakyReLU(alpha=0.1)),
    Dropout(0.2),  # Adjust dropout rate
    Dense(1)  # Linear regression output layer
])

# Adjusted Training Parameters
optimizer = Adam(learning_rate=0.0001)  # Adjust learning rate
model.compile(optimizer=optimizer, loss='mean_squared_error')
model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss = model.evaluate(X_test_scaled, y_test)
print("Test Loss:", test_loss)

# Make predictions
predictions = model.predict(X_test_scaled)

# Print some predictions and actual values
print("Some Predictions and Actual Values:")
for i in range(10):
    print("Predicted Price:", predictions[i][0], "Actual Price:", y_test.iloc[i])