import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load the dataset
data = pd.read_csv('fashion-MNIST.csv')

# Split features and labels
X = data.iloc[:, 1:].values.astype('float32')  # Features (pixels)
y = data.iloc[:, 0].values.astype('int32')     # Labels (categories)

# Preprocess the data
X /= 255.0  # Normalize pixel values to be between 0 and 1
X = X.reshape(-1, 28, 28, 1)  # Reshape the features into 28x28 grayscale images

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
num_classes = len(np.unique(y))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

# Make predictions
predictions = model.predict(X_test)

# Print some actual and predicted classes
print("Some actual and predicted classes:")
for i in range(10):  # Print predictions for the first 10 images
    actual_class = np.argmax(y_test[i])
    predicted_class = np.argmax(predictions[i])
    print(f"Sample {i+1}: Actual cloth class: {actual_class}, Predicted cloth class: {predicted_class}")