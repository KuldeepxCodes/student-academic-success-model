#  Let's Project 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

# Load Data
df = pd.read_csv('ml_csv.csv')

# Encode categorical columns
le = LabelEncoder()
df['Internet'] = le.fit_transform(df["Internet"])
df['Passed'] = le.fit_transform(df["Passed"])

# Features
features = ['StudyHours','Attendance','PastScore','SleepHours']

# Scaling
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

X = df[features] 
y = df['Passed']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fail","Pass"],
            yticklabels=["Fail","Pass"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

print("------ Predict Your Result -----")

try:
    study_hours = float(input("Enter your study hours: "))
    attendance = float(input("Enter your attendance: "))
    past_score = float(input("Enter your past score: "))
    sleep_hours = float(input("Enter your sleep hours: "))

    user_input = pd.DataFrame([{
        'StudyHours': study_hours,
        'Attendance': attendance,
        'PastScore': past_score,
        'SleepHours': sleep_hours
    }])

    user_input_scaled = scaler.transform(user_input)

    prediction = model.predict(user_input_scaled)[0]

    result = "Pass" if prediction == 1 else "Fail"

    print(f"Prediction based on input: {result}")

except Exception as e:
    print("An error occurred:", e)
