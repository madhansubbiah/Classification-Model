import streamlit as st
import pandas as pd
import pickle
import json

# Load your trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)  # Load the model

# Load feature values from JSON files
with open('user_type.json', 'r') as json_file:
    user_types = json.load(json_file)['USER_TYPE']

with open('protocol_nm.json', 'r') as json_file:
    protocol_names = json.load(json_file)['PROTOCOL_NM']

with open('protocol_status.json', 'r') as json_file:
    protocol_statuses = json.load(json_file)['PROTOCOL_STATUS']

with open('country.json', 'r') as json_file:
    countries = json.load(json_file)['COUNTRY']

with open('country_status.json', 'r') as json_file:
    country_statuses = json.load(json_file)['COUNTRY_STATUS']

with open('site_status.json', 'r') as json_file:
    site_statuses = json.load(json_file)['SITE_STATUS']

with open('user_status.json', 'r') as json_file:
    user_statuses = json.load(json_file)['USER_STATUS']

# Function to make predictions
def predict(email_address, site_no, user_type, protocol_nm, protocol_status, country, country_status, site_status, user_status):
    # Prepare the input data for the model
    input_data = pd.DataFrame({
        'SITE_NO': [site_no],  # User input
    })
    
    # One-Hot Encoding for categorical inputs
    user_type_encoded = pd.get_dummies(pd.Series([user_type]), prefix='USER_TYPE')
    protocol_status_encoded = pd.get_dummies(pd.Series([protocol_status]), prefix='PROTOCOL_STATUS')
    country_encoded = pd.get_dummies(pd.Series([country]), prefix='COUNTRY')
    country_status_encoded = pd.get_dummies(pd.Series([country_status]), prefix='COUNTRY_STATUS')
    site_status_encoded = pd.get_dummies(pd.Series([site_status]), prefix='SITE_STATUS')
    user_status_encoded = pd.get_dummies(pd.Series([user_status]), prefix='USER_STATUS')

    # Concatenate all encoded features into the input DataFrame
    input_data = pd.concat([input_data, user_type_encoded, protocol_status_encoded, country_encoded,
                            country_status_encoded, site_status_encoded, user_status_encoded], axis=1)

    # Align the input DataFrame with the model's expected feature names
    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    # Make prediction
    prediction = model.predict(input_data)
    
    return prediction[0]  # Return the first prediction

# Streamlit app layout
st.title("Provisioning Prediction App")

# User input
email_address = st.text_input("Enter Email Address (EMAIL_ADDRESS):")
site_no = st.text_input("Enter Site Number (SITE_NO):")

# Dropdowns for user selections
user_type = st.selectbox("Select User Type (USER_TYPE):", user_types)
protocol_nm = st.selectbox("Select Protocol Name (PROTOCOL_NM):", protocol_names)
protocol_status = st.selectbox("Select Protocol Status (PROTOCOL_STATUS):", protocol_statuses)
country = st.selectbox("Select Country (COUNTRY):", countries)
country_status = st.selectbox("Select Country Status (COUNTRY_STATUS):", country_statuses)
site_status = st.selectbox("Select Site Status (SITE_STATUS):", site_statuses)
user_status = st.selectbox("Select User Status (USER_STATUS):", user_statuses)

# Button to make prediction
if st.button("Predict"):
    if email_address and site_no:
        result = predict(email_address, site_no, user_type, protocol_nm, protocol_status, country, country_status, site_status, user_status)
        if result == 0:
            st.success("Not Provisioned to Inform")
        else:
            st.success("Provisioned to Inform")
        
        # Display the user inputs for reference
        st.write(f"Email Address: {email_address}")
        st.write(f"Site Number: {site_no}")
        st.write(f"Protocol Name: {protocol_nm}")
    else:
        st.error("Please fill in all fields.")
