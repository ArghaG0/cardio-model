import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_screening_data(filepath):
    df=pd.read_csv(filepath,sep=';')

    df['age_years']=df['age']/365.25
    df=df.drop(['age','id'],axis=1)

    features = ['age_years','gender','height','weight','ap_hi','ap_lo','cholesterol','gluc','smoke','alco','active']
    target='cardio'

    x=df[features]
    y=df[target]

    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(x)

    X_scaled_df=pd.DataFrame(X_scaled,columns=features)
    return X_scaled_df,y

X_data,y_labels=prepare_screening_data('../data/cardio_train.csv')
print(X_data.head())

X_data.to_csv('normalized_cardio_data.csv',index=False)
y_labels.to_csv('cardio_target_labels.csv',index=False)