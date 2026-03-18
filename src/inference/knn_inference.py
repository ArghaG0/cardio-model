#knn_inference.py
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np

def screen_patient(new_patient_data,tree_path,labels_path, original_data_path):
    with open(tree_path,'rb') as file:
        kd_tree = pickle.load(file)
    
    labels_df=pd.read_csv(labels_path)
    y_labels=labels_df['cardio'].values

    raw_df=pd.read_csv(original_data_path, sep=';')
    raw_df['age_years']=raw_df['age']/365.25
    features=['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    X_train_raw=raw_df[features]

    scaler=StandardScaler()
    scaler.fit(X_train_raw)

    new_patient_df=pd.DataFrame([new_patient_data],columns=features)
    new_patient_scaled=scaler.transform(new_patient_df)

    distances,indices=kd_tree.kneighbors(pd.DataFrame(new_patient_scaled,columns=features))

    neighbor_indices=indices[0]
    neighbor_risks=y_labels[neighbor_indices]

    risk_score=np.mean(neighbor_risks)*100

    return risk_score,neighbor_indices

dummy_patient = [30.0, 1, 165, 60.0, 110, 70, 1, 1, 0, 0, 1]

risk_percentage,similar_patients=screen_patient(dummy_patient,'../../models/knn_model.pkl','../../data/cardio_target_labels.csv','../../data/cardio_train.csv')
print("Cardiovascular Risk Probability:",risk_percentage,"%")
print("Patients at indices:",similar_patients)