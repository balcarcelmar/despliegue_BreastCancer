import streamlit as st
import pandas as pd
import joblib

# Load the scaler, model, and label encoder
scaler = joblib.load('minmax_scaler.pkl')
loaded_model = joblib.load('best_rf_model.joblib')
label_encoder = joblib.load('label_encoder.pkl')

# Define the features that the user will input
user_input_features = [
    'radius_mean', 'texture_mean', 'smoothness_mean',
    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'
]

# Define the min/max values for each feature based on the user's input
feature_ranges = {
    'radius_mean': {'min': 0.1, 'max': 50.0},
    'texture_mean': {'min': 0.01, 'max': 100.0},
    'smoothness_mean': {'min': 0.0001, 'max': 0.3},
    'concave points_mean': {'min': 0.0, 'max': 0.5},
    'symmetry_mean': {'min': 0.5, 'max': 2},
    'fractal_dimension_mean': {'min': 0.0001, 'max': 0.3}
}

# Streamlit App Title
st.title('Breast Cancer Prediction App')
st.write('Enter the values for the following features to get a prediction:')

# Create input fields for each feature
input_data = {}
for feature in user_input_features:
    min_val = feature_ranges[feature]['min']
    max_val = feature_ranges[feature]['max']
    default_val = (min_val + max_val) / 2 # Set default to midpoint for better UX
    input_data[feature] = st.slider(f'**{feature.replace("_", " ").title()}**', min_val, max_val, default_val)

# Create a DataFrame from the input data
input_df = pd.DataFrame([input_data])

st.subheader('Input Features:')
st.write(input_df)

# Scale the input features
scaled_input = scaler.transform(input_df)
scaled_input_df = pd.DataFrame(scaled_input, columns=user_input_features)

st.subheader('Scaled Input Features:')
st.write(scaled_input_df)

# Make prediction when button is pressed
if st.button('Predict'):
    prediction = loaded_model.predict(scaled_input_df)
    decoded_prediction = label_encoder.inverse_transform(prediction)

    st.subheader('Prediction Result:')
    if decoded_prediction[0] == 'B':
        st.success('The tumor is predicted to be **Benign** (No Cáncer).')
    else:
        st.error('The tumor is predicted to be **Malignant** (Cáncer).')
