import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Clear any existing sessions to prevent conflicts
tf.keras.backend.clear_session()

# Rebuild model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(12,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model (required before loading weights)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load weights
try:
    model.load_weights('model.weights.h5')
    st.success("Model weights loaded successfully!")
except Exception as e:
    st.error(f"Error loading model weights: {e}")

# Load encoders and scaler
try:
    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    st.success("Encoders and scaler loaded successfully!")
except Exception as e:
    st.error(f"Error loading encoders/scaler: {e}")

# Streamlit app UI
st.title('Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Add a predict button
if st.button('Predict Churn'):
    try:
        # Prepare numeric + label-encoded input
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })

        # Transform Geography with proper feature name
        geo_df = pd.DataFrame([[geography]], columns=['Geography'])
        geo_encoded = onehot_encoder_geo.transform(geo_df).toarray()
        geo_encoded_df = pd.DataFrame(
            geo_encoded,
            columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
        )

        # Merge encoded geography with other features
        full_input = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

        # Ensure the correct order of columns (this is crucial)
        # You may need to adjust this based on your training data column order
        expected_columns = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 
                          'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'] + \
                         list(onehot_encoder_geo.get_feature_names_out(['Geography']))
        
        full_input = full_input.reindex(columns=expected_columns, fill_value=0)

        # Scale the input
        input_scaled = scaler.transform(full_input)

        # Predict
        prediction = model.predict(input_scaled, verbose=0)
        prediction_proba = prediction[0][0]

        st.write(f'🔎 **Churn Probability:** `{prediction_proba:.2f}`')

        if prediction_proba > 0.5:
            st.error('⚠️ The customer is **likely to churn**.')
        else:
            st.success('✅ The customer is **not likely to churn**.')
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.write("Debug info:")
        st.write(f"Input shape: {full_input.shape}")
        st.write(f"Expected input shape: (1, 12)")
        st.write("Columns:", full_input.columns.tolist())