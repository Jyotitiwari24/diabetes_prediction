import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/diabetes.csv")

# Split features and target
X = df.drop(columns=['Outcome'])
Y = df['Outcome']

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, stratify=Y, random_state=2
)

# Train model
model = svm.SVC(kernel='linear')
model.fit(X_train, Y_train)

# Evaluate
train_acc = accuracy_score(model.predict(X_train), Y_train)
test_acc = accuracy_score(model.predict(X_test), Y_test)

print(f"Train Accuracy: {train_acc}")
print(f"Test Accuracy: {test_acc}")

# Save model + scaler
joblib.dump(model, "model/diabetes_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("Model and scaler saved!")
