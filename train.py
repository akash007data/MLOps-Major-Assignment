# train.py - trains DecisionTreeClassifier on Olivetti dataset

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

# Load dataset
data = fetch_olivetti_faces()
X = data.data
y = data.target

# Split: 70% train, 30% test (as required)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
model_path = "models/savedmodel.pth"
joblib.dump(clf, model_path)

print("Training completed. Model saved at:", model_path)
