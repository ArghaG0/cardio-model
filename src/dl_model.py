import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

raw_df = pd.read_csv('../data/cardio_train.csv', sep=';')
raw_df = raw_df[(raw_df['ap_hi'] >= 90) & (raw_df['ap_hi'] <= 200)]
raw_df = raw_df[(raw_df['ap_lo'] >= 60) & (raw_df['ap_lo'] <= 120)]
raw_df['age_years'] = raw_df['age'] / 365.25
raw_df['bmi'] = raw_df['weight'] / ((raw_df['height'] / 100) ** 2)

features = ['age_years', 'gender', 'bmi', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
X = raw_df[features].values
y = raw_df['cardio'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

class CardioNet(nn.Module):
    def __init__(self):
        super(CardioNet, self).__init__()
        self.layer1 = nn.Linear(10, 32)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x

model = CardioNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
print("Starting Training...")

for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} completed. Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    predicted_classes = (test_predictions >= 0.5).float()

accuracy = accuracy_score(y_test, predicted_classes.numpy()) * 100
precision = precision_score(y_test, predicted_classes.numpy()) * 100
recall = recall_score(y_test, predicted_classes.numpy()) * 100

print("--- Deep Learning Results ---")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")

model_save_path = '../models/cardio_net_v1.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model successfully saved to {model_save_path}")