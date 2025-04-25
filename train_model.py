import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load dataset
data = pd.read_csv('german_credit_data.csv')

# Handle missing values
data['Saving accounts'].fillna(data['Saving accounts'].mode()[0], inplace=True)
data['Checking account'].fillna(data['Checking account'].mode()[0], inplace=True)
data['Credit amount'].fillna(data['Credit amount'].median(), inplace=True)
data['Duration'].fillna(data['Duration'].median(), inplace=True)

# Label Encoding
le = LabelEncoder()
for col in ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']:
    data[col] = le.fit_transform(data[col])

# Feature Engineering - Credit to Duration Ratio
data['Credit_Duration_Ratio'] = data['Credit amount'] / data['Duration']

# Create Target Variable
data['Class'] = data['Credit amount'].apply(lambda x: 1 if x > 5000 else 0)

# Define features and target
X = data.drop(columns=['Unnamed: 0', 'Class'], errors='ignore')
y = data['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluation
y_pred = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature importance
feat_importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=feat_importance, y=feat_importance.index)
plt.title("Feature Importance")
plt.show()

# Save model
joblib.dump(rf_model, 'credit_risk_model.pkl')
print("Model saved as 'credit_risk_model.pkl'.")
