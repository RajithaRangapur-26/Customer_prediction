import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("===== CUSTOMER CHURN PREDICTION SYSTEM =====")

# Load dataset (Make sure CSV is in same folder)
try:
    data = pd.read_csv("Telco-Customer-Churn.csv")
except:
    print("❌ ERROR: Dataset file not found!")
    print("👉 Keep 'Telco-Customer-Churn.csv' in same folder")
    exit()

print("\n✅ Dataset Loaded Successfully!")

# Display basic info
print("\nFirst 5 Rows:")
print(data.head())

print("\nDataset Info:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

# Drop unnecessary column
if 'customerID' in data.columns:
    data.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

# Convert target column
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

# Encode categorical variables
data = pd.get_dummies(data, drop_first=True)

print("\n✅ Data Preprocessing Completed")

# Split features and target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("\n✅ Model Trained Successfully!")

# Prediction
y_pred = model.predict(X_test)

print("\nPredicted Values (First 10):")
print(y_pred[:10])

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\n🎯 Accuracy:", round(accuracy * 100, 2), "%")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Plot feature importance
plt.figure()
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
plt.title("Top Features")
plt.show()

# Sample prediction
sample = X.iloc[[0]]
prediction = model.predict(sample)

print("\nSample Customer Result:")
if prediction[0] == 1:
    print("👉 Customer will CHURN")
else:
    print("👉 Customer will STAY")
