from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler 
import pickle

# Training three models to predict heart failure

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df.drop(['time'], axis=1, inplace=True)
df.drop(df[df['ejection_fraction']==80].index, axis=0, inplace=True)

y = df['DEATH_EVENT']
X = df.drop('DEATH_EVENT', axis=1)

scale_list = ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium']
scaler = StandardScaler()
X[scale_list] = scaler.fit_transform(X[scale_list])


# Logistic Regresion, SVC, XGBoost

Logi = LogisticRegression(C=0.46415, penalty='l2')
Logi.fit(X,y)
pickle.dump(Logi,open('Logistic_HF.pkl', 'wb'))


SVM = SVC(kernel='linear', probability=True)
SVM.fit(X,y)
pickle.dump(SVM, open('SVC_HF.pkl', 'wb'))


xgb_params = {'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 300}
XGB = XGBClassifier(**xgb_params)
XGB.fit(X,y)
pickle.dump(XGB,open('XGB_HF.pkl', 'wb'))
