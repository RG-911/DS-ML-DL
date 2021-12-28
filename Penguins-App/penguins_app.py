# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.ensemble import RandomForestClassifier


# START BUILDING WEB APP
st.write("""
#PENGUIN PREDICTION APP

This app precits **Palmer Penguin** species!

""")

st.sidebar.header("USER INPUT FEATURES")
st.sidebar.markdown("""
[Example CSV input file] (https://github.com/RG-911/Penguins-App/raw/main/penguins_cleaned.csv)
""")

upload_file = st.sidebar.file_uploader('Upload your input csv file', type=['csv'])
if upload_file is not None:
    input_df = pd.read_csv(upload_file)
else: 
    def user_input():
        island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('Sex', ('male', 'female'))
        bill_length_mm = st.sidebar.slider('Bill lenght (mm)', 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider('Bill depht (mm)', 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider('Flipper lenght (mm)', 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0, 6300.0, 4200.0 )

        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm':bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input()

# COMBINE USER INPUT DATA WITH ENTIRE PENGUINS DATASET
penguins_raw = pd.read_csv('penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df, penguins], axis=0)

# ENCODING CATEGORICAL PARAMETERS
encode = ['sex', 'island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df=pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

#DISPLAY USER INPUT FEATURES
st.subheader('User Input Features')

if upload_file is not None:
    st.write(df)
else:
    st.write('Awating CSV file to be uploaded, Currently using example input parameters (shown below)')
    st.write(df)

# LOAD SAVED CLASSIFICATION MODEL
load_rfc = pickle.load(open('penquins_rfc.pkl', 'rb'))

#APPLYING MODEL TO MAKE PREDICTIONS
prediction = load_rfc.predict(df)
prediction_proba = load_rfc.predict_proba(df)

st.subheader('Prediction')
penguins_species = np.array(['Adelie', 'Chinstrap','Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)