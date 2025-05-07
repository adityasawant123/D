import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load the dataset
file_path = "letter-recognition.csv"
data = pd.read_csv(file_path)

# Check the column names
print("Column Names:", data.columns)

# Preprocess the data
X = data.drop(columns=['T'])  # Features
y = data['T']  # Target

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')  # Output layer with number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)

# Make predictions
predictions = model.predict(X_test)

# Decode the predicted labels
predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

# Decode the actual labels
actual_labels = label_encoder.inverse_transform(y_test)

# Print some predicted and actual results
print("Some Predicted and Actual Results:")
for i in range(10):
    print("Predicted:", predicted_labels[i], "Actual:", actual_labels[i])