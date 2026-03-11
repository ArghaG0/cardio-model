#model_evaluation.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

raw_df = pd.read_csv('../data/cardio_train.csv', sep=';')

raw_df = raw_df[(raw_df['ap_hi'] >= 90) & (raw_df['ap_hi'] <= 200)]
raw_df = raw_df[(raw_df['ap_lo'] >= 60) & (raw_df['ap_lo'] <= 120)]

raw_df['age_years'] = raw_df['age'] / 365.25
raw_df['bmi'] = raw_df['weight'] / ((raw_df['height'] / 100) ** 2)

features = ['age_years', 'gender', 'bmi', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

X = raw_df[features]
y = raw_df['cardio']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=50, algorithm='kd_tree')
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred) * 100
recall = recall_score(y_test, y_pred) * 100

print("Accuracy:", accuracy, "%")
print("Precision:", precision, "%")
print("Recall:", recall, "%")