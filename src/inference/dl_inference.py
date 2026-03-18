#df_inference.py
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

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

def screen_patient_dl(patient_data, model_path, original_data_path):
    raw_df = pd.read_csv(original_data_path, sep=';')
    raw_df = raw_df[(raw_df['ap_hi'] >= 90) & (raw_df['ap_hi'] <= 200)]
    raw_df = raw_df[(raw_df['ap_lo'] >= 60) & (raw_df['ap_lo'] <= 120)]
    raw_df['age_years'] = raw_df['age'] / 365.25
    raw_df['bmi'] = raw_df['weight'] / ((raw_df['height'] / 100) ** 2)
    
    features = ['age_years', 'gender', 'bmi', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    X_train_raw = raw_df[features].values
    
    scaler = StandardScaler()
    scaler.fit(X_train_raw)

    age_years, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active = patient_data
    bmi = weight / ((height / 100) ** 2)
    processed_patient = [[age_years, gender, bmi, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active]]
    
    patient_scaled = scaler.transform(processed_patient)
    patient_tensor = torch.FloatTensor(patient_scaled)

    model = CardioNet()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        prediction = model(patient_tensor)
        risk_score = prediction.item() * 100

    return risk_score

patient_55_unhealthy = [55.0, 1, 170, 85.0, 150, 95, 3, 2, 1, 0, 1]
patient_30_healthy = [30.0, 1, 165, 60.0, 110, 70, 1, 1, 0, 0, 1]

risk_55 = screen_patient_dl(patient_55_unhealthy, '../../models/cardio_net_v1.pth', '../../data/cardio_train.csv')
risk_30 = screen_patient_dl(patient_30_healthy, '../../models/cardio_net_v1.pth', '../../data/cardio_train.csv')

print(f"55-Year-Old At-Risk Patient: {risk_55:.2f}% Probability")
print(f"30-Year-Old Healthy Patient: {risk_30:.2f}% Probability")