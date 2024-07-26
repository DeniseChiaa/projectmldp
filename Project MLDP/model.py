import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor

# Load the model
with open("gbm_model.pkl", 'rb') as file:
    load_clf = pickle.load(file)

st.write("""
# Real Estate Price Prediction
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    property_type = st.sidebar.selectbox('Property Type', ('Flat', 'House', 'Penthouse', 'Farm House', 'Lower Portion', 'Upper Portion', 'Room'))
    city = st.sidebar.selectbox('City', ('Islamabad', 'Karachi', 'Lahore', 'Faisalabad', 'Rawalpindi'))
    province_name = st.sidebar.selectbox('Province', ('Islamabad Capital', 'Sindh', 'Punjab'))
    latitude = st.sidebar.slider('Latitude', 0.000000, 100.000000, 24.972287	)
    longitude = st.sidebar.slider('Longitude', 0.000000, 100.000000, 67.066298)
    baths = st.sidebar.slider('Bathrooms', 0, 10, 3)
    area_size = st.sidebar.slider('Area Size', 0.0, 10.0, 3.2)
    bedrooms = st.sidebar.slider('Bedrooms', 0, 10, 4)
    area_type = st.sidebar.selectbox('Area Type', ('Marla', 'Kanal'))
    area_category = st.sidebar.selectbox('Area Category', (
        '0-5 Marla', '5-10 Marla', '1-5 Kanal', '10-15 Marla', '15-20 Marla',
        '10-15 Kanal', '15-20 Kanal', '20-30 Kanal', '30-40 Kanal', '40-50 Kanal',
        '50-60 Kanal', '60-70 Kanal', '70-80 Kanal', '80-90 Kanal', '90-100 Kanal',
        '100-200 Kanal', '200-300 Kanal', '400-500 Kanal', '500-600 Kanal', 
        '600-700 Kanal', '700-800 Kanal'
    ))
    
    purpose = st.sidebar.selectbox('Purpose', ('Sale', 'Rent'))
    area = st.sidebar.slider('Area', 0, 1000, 194)

    data = {
        'property_type': property_type,
        'city': city,
        'province_name': province_name,
        'latitude': latitude,
        'longitude': longitude,
        'baths': baths,
        'area_size': area_size,
        'bedrooms': bedrooms,
        'area_type': area_type,
        'area_category': area_category,
        'purpose': purpose,
        'area': area
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

# One-hot encode categorical features
df_encoded = pd.get_dummies(df)

# Load the model's feature names from an external file
try:
    with open("feature_names.pkl", 'rb') as file:
        feature_names = pickle.load(file)
except FileNotFoundError:
    st.error("Feature names file not found. Ensure that feature names are saved during training.")
    st.stop()

# Align with model's expected features
for feature in feature_names:
    if feature not in df_encoded.columns:
        df_encoded[feature] = 0  # Add missing feature as a column with 0 value

df_encoded = df_encoded[feature_names]  # Reorder columns to match the model's expected order

# Prediction
prediction = load_clf.predict(df_encoded)

st.subheader('Prediction')
st.write(f'Estimated Price: {prediction[0]*-1}')
