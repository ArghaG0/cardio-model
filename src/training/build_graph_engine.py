#build_graph_engine.py
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle

def build_kd_tree(data_path):
    df = pd.read_csv(data_path)
    
    knn_model = NearestNeighbors(n_neighbors=10, algorithm='kd_tree')
    knn_model.fit(df)
    
    with open('../models/patient_similarity_tree.pkl', 'wb') as file:
        pickle.dump(knn_model, file)
        
    return knn_model

tree_model = build_kd_tree('../data/normalized_cardio_data.csv')
print("KD-Tree built and saved successfully!")