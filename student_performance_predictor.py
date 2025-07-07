
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

try:
    user_input = float(input("Enter number of hours studied: "))
except ValueError:
    print("❌ Invalid input. Please enter a number.")
    exit()

data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Passed':        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

X = df[['Hours_Studied']]
y = df['Passed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

user_input_array = pd.DataFrame([[user_input]], columns=["Hours_Studied"])
result = model.predict(user_input_array)[0]
probability = model.predict_proba(user_input_array)[0][1]
print("\nPrediction:", "✅ Pass" if result == 1 else "❌ Fail", f"(Confidence: {probability:.2f})")

y_pred = model.predict(X_test)

correct = accuracy_score(y_test, y_pred) * 100
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()

print("\n=== Simple Model Evaluation ===")
print(f"✅ The model predicted correctly {correct:.0f}% of the time.")



X_plot = np.linspace(0, 10, 100).reshape(-1, 1)
y_plot = model.predict_proba(X_plot)[:,1]

plt.figure(figsize=(8, 5))
plt.plot(X_plot, y_plot, color='green', label='Probability Curve')
plt.scatter(df['Hours_Studied'], df['Passed'], color='blue', label='Original Data')
plt.scatter(user_input, result, color='red', s=100, label=f'Your Input ({user_input} hrs)')

plt.annotate(f"{'Pass' if result == 1 else 'Fail'} ({probability:.2f})",
             (user_input, result),
             textcoords="offset points",
             xytext=(10, 10),
             ha='left',
             color='red',
             fontsize=10)

plt.xlabel('Hours Studied')
plt.ylabel('Probability / Outcome')
plt.title('Logistic Regression: Study Hours vs Pass/Fail')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
