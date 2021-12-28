import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


st.write("""
# MOBILE OS PREDICTION APP

The app will help users to detect the OS of a smartphone.

""")

st.sidebar.header('**INPUT PARAMETRS**')

def set_input_parameters():
    model = st.sidebar.text_input('Model')
    storage = st.sidebar.selectbox('Storage', ('512MB', '8GB', '16GB', '32GB', '64GB', '128GB', '256GB', '512GB', '1TB'))
    colour = st.sidebar.text_input('Colour')
   

    parameters = {'model': model,
                'colour': colour,
                'storage': storage
                }

    features = pd.DataFrame(parameters, index=[0])
    return features

df = set_input_parameters()

st.subheader('Smartphone parameters')
st.write(df)


path = '/Users/LAMAN/Google Drive/Data Science/DataProfessor/smartphones.csv'
smartphones = pd.read_csv(path)
smartphones.dropna(inplace=True)
colour = {'schwarz': 'black', 'silber': 'silver', 'blau': 'blue', 'champagner':'champagne','violett': 'violet', 'grau': 'gray',
         'koralle':'coral',  'mandarinrot':'mandarin red', 'rot':'red', 'kupfer': 'copper', 'anthrazit':'anthracite',
         'weiss': 'white','space grau': 'space gray', 'spacegrau': 'space gray', 'gelb': 'yellow', 'schwarz graphit': 'black graphite',
         'graphit': 'graphite', 'pazifikblau': 'pacific blue', 'hellgrau': 'light gray'}

smartphones.replace({'Colour': colour}, inplace=True)

#Split the data
X = smartphones.drop(['Brand','Operating System'], axis =1)
y = smartphones['Operating System']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)

#Random Forest Classification
rfc_model = make_pipeline(OneHotEncoder(handle_unknown='ignore'), RandomForestClassifier())

n_estimators_range = np.arange(10,210,10)
param_grid = dict( randomforestclassifier__n_estimators=n_estimators_range)
grid = GridSearchCV(estimator=rfc_model, param_grid=param_grid, cv=5)

grid.fit(X_train, y_train)

st.subheader('The predicted Mobile OS')
rfc_model_pred = grid.predict(df)
st.write(rfc_model_pred)

#probability = grid.predict_proba(df)

#st.subheader('Model evaluation parameters')
#st.write(probability)