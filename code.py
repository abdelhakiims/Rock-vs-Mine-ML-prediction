import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load dataset — place sonar.csv inside a /data folder in your project directory
# header=None because the dataset has no column name row, just raw data
sonar_data = pd.read_csv('data/sonar.csv', header=None)

# Separate features (X) and labels (Y)
# Column 60 is the label: 'M' for Mine, 'R' for Rock
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

# Split into train/test sets
# test_size=0.1 means 10% of data is used for testing
# stratify=Y ensures both splits have a balanced proportion of M and R
# random_state=1 freezes the random split so results are reproducible
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=1
)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluate on training data
train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_predictions)
print(f"Training accuracy: {train_accuracy:.4f}")

# Evaluate on test data
test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_predictions)
print(f"Test accuracy:     {test_accuracy:.4f}")

# --- Predict a single sample ---
# This is one data point taken from the dataset to test the model manually
input_data = (0.0223,0.0375,0.0484,0.0475,0.0647,0.0591,0.0753,0.0098,0.0684,0.1487,
              0.1156,0.1654,0.3833,0.3598,0.1713,0.1136,0.0349,0.3796,0.7401,0.9925,
              0.9802,0.8890,0.6712,0.4286,0.3374,0.7366,0.9611,0.7353,0.4856,0.1594,
              0.3007,0.4096,0.3170,0.3305,0.3408,0.2186,0.2463,0.2726,0.1680,0.2792,
              0.2558,0.1740,0.2121,0.1099,0.0985,0.1271,0.1459,0.1164,0.0777,0.0439,
              0.0061,0.0145,0.0128,0.0145,0.0058,0.0049,0.0065,0.0093,0.0059,0.0022)

input_array = np.asarray(input_data).reshape(1, -1)
prediction = model.predict(input_array)

print(f"\nSample prediction: {'Mine' if prediction[0] == 'M' else 'Rock'}")