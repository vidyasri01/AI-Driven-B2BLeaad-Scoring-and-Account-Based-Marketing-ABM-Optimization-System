import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (replace 'data.csv' with your actual dataset file)
# Ensure the dataset contains no missing values and correct column names
data = pd.read_csv('data.csv')

# Check for missing values
if data.isnull().sum().any():
    raise ValueError("Dataset contains missing values. Please clean the data before proceeding.")

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Features (independent variables) and target (dependent variable)
# Replace 'Feature1' and 'Feature2' with your actual feature column names
# Replace 'Purchased' with your actual target column
try:
    X = data[['Feature1', 'Feature2']]
    y = data['Purchased']
except KeyError as e:
    raise KeyError(f"Column not found: {e}. Ensure the dataset has the correct column names.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Predict the probability of a new data point (example: [50, 85])
try:
    new_data = np.array([[50, 85]])  # Replace with your feature values
    predicted_prob = model.predict_proba(new_data)[0][1]
    print(f"\nPredicted Probability of Purchase: {predicted_prob:.2f}")
except ValueError as e:
    print(f"Error in prediction: {e}. Ensure the input matches the feature dimensions.")
