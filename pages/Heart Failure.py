import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")


st.write("""
# Heart Failure Prediction App

""")

st.sidebar.header('User Input Features')


# Collects user input features into dataframe

def user_input_features():

    model = st.sidebar.selectbox('Model',('Logistic Regression','SVC','XGBoost'))

    age = st.sidebar.slider('Age', 50,90,70)
    serum_creatinine = st.sidebar.slider('Level of creatinine in the blood (mg/dL)', .7,2.0,1.1)
    serum_sodium = st.sidebar.slider('Level of sodium in the blood (mEq/L)', 110,150,130)
    eject_frac = st.sidebar.slider('Percentage of blood leaving per contraction', 10,70,40)
    platelets = st.sidebar.slider('Platelets in the blood (kiloplatelets/mL)', 100000,800000,300000)
    cpk = st.sidebar.slider('Creatinine phosphokinase level (mcg/L)', 0,2000,250)
    
    sex = st.sidebar.selectbox('Sex',('Male','Female'))
    anemic = st.sidebar.selectbox('Anemic',('No','Yes'))
    diabetic = st.sidebar.selectbox('Diabetic',('No','Yes'))
    blood_pressure = st.sidebar.selectbox('High blood pressure',('No','Yes'))
    smoking = st.sidebar.selectbox('Smoker',('No','Yes'))


    data = {'age': age, #
            'anaemia': anemic,
            'creatinine_phosphokinase': cpk,
            'diabetes': diabetic,
            'ejection_fraction': eject_frac, #
            'high_blood_pressure': blood_pressure,
            'platelets': platelets,
            'serum_creatinine': serum_creatinine, 
            'serum_sodium': serum_sodium, #
            'sex': sex,
            'smoking': smoking
            }  

    features = pd.DataFrame(data, index=[0])
    return features, model

df, model = user_input_features()

# Encode inputs
encode = ['anaemia','diabetes','high_blood_pressure','smoking']
cat_mapping = {'Yes': 1, 'No': 0 }
for col in encode:
    df[col] = df[col].map(cat_mapping)

df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})

# Displays the user input features
st.subheader('User Input features')

st.write(df.iloc[:,:5])
st.write(df.iloc[:,5:])
# Save column names for later
columns_list = list(df.columns)

# Reading the original dataset in to scale the input
HF_train = pd.read_csv('heart_failure_clinical_records_dataset.csv')
HF_train.drop(['time'], axis=1, inplace=True)

HF_train = HF_train.drop('DEATH_EVENT', axis=1)

df = pd.concat([df,HF_train],axis=0)

scale_list = ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium']
scaler = StandardScaler()
df[scale_list] = scaler.fit_transform(df[scale_list])

# Getting back just the user input
df = df[:1]

# Load in trained models

if model == 'Logistic Regression':
    load_clf =  pickle.load(open('Logistic_HF.pkl', 'rb'))
elif model == 'SVC':
    load_clf =  pickle.load(open('SVC_HF.pkl', 'rb'))
elif model == 'XGBoost':
    load_clf =  pickle.load(open('XGB_HF.pkl', 'rb'))

prediction = load_clf.predict(df)


# Prediction Probabilities

st.write('#')
st.subheader('Mortality Prediction')
DEATH_EVENT = np.array(['NO','YES'])
st.write(DEATH_EVENT[prediction])

prediction_proba = load_clf.predict_proba(df)
prediction_proba = np.round(prediction_proba,4)
proba_df = pd.DataFrame(prediction_proba)

print(proba_df)

st.subheader('Mortality Event Probabilities')
st.dataframe(proba_df.style.format("{:.2%}"))



# Feature importances

if model == 'XGBoost':
    
    st.subheader('XGB Feature Importance')

    imp_df = pd.DataFrame(list(zip(columns_list, load_clf.feature_importances_)), columns = ['Feature','Importance'])
   
    fig, ax = plt.subplots()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    sns.barplot(x="Feature", y="Importance", data=imp_df)
    st.pyplot(fig)

    st.write(imp_df)

elif model == 'Logistic Regression':
    st.subheader('Regression Coefficients')
   
    imp_df = pd.DataFrame(list(zip(columns_list, load_clf.coef_[0])), columns = ['Feature','Coefficient'])
   
    fig, ax = plt.subplots()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    sns.barplot(x="Feature", y="Coefficient", data=imp_df)
    st.pyplot(fig)

    st.write(imp_df)

elif model == 'SVC':
    st.subheader('SVC Coefficients')
   
    imp_df = pd.DataFrame(list(zip(columns_list, load_clf.coef_[0])), columns = ['Feature','Coefficient'])
   
    fig, ax = plt.subplots()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    sns.barplot(x="Feature", y="Coefficient", data=imp_df)
    st.pyplot(fig)

    st.write(imp_df)

