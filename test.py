# test.py - loads the saved model and evaluates accuracy

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = fetch_olivetti_faces()
X = data.data
y = data.target

# Same split as train.py
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Load model
clf = joblib.load("models/savedmodel.pth")

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Test Accuracy:", acc)
