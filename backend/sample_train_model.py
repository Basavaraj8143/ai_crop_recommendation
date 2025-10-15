# backend/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# --- Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "crop_recommendation.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "crop_model.pkl")

# --- Load dataset
df = pd.read_csv(DATA_PATH)

# --- Split features and labels
X = df.drop('label', axis=1)
y = df['label']

# --- Train-test split (you can change test_size if you want)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Model evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

# --- Save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)

# --- Output results
print("✅ Model trained and saved at:", MODEL_PATH)
print(f"📊 Training data size: {len(X_train)} samples")
print(f"🧪 Testing data size: {len(X_test)} samples")
print(f"🎯 Model Accuracy: {accuracy:.2f}%")

