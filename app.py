import streamlit as st
import pickle #import joblib
import pandas as pd

# Load the trained model
model = joblib.load('delivery_time_model.pkl')  # Adjust the path accordingly

# Title of the app
st.title('Delivery Time Prediction')

# Input fields for order details
product_category = st.selectbox('Product Category', ['Category A', 'Category B', 'Category C'])
customer_location = st.text_input('Customer Location')
shipping_method = st.selectbox('Shipping Method', ['Standard', 'Express', 'Same Day'])

# Button for prediction
if st.button('Predict Delivery Time'):
    # Gather input data into a DataFrame
    input_data = pd.DataFrame({
        'product_category': [product_category],
        'customer_location': [customer_location],
        'shipping_method': [shipping_method]
    })
    
    # Make prediction using the model
    prediction = model.predict(input_data)

    # Display the prediction
    st.success(f'Estimated Delivery Time: {prediction[0]} days')
