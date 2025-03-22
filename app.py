import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Timelytics - Delivery Time Prediction",
    page_icon=":truck:",
    layout="wide"
)

# Title and description
st.title("Timelytics: Delivery Time Prediction")

st.markdown(
    """
    ### Optimize your supply chain with advanced forecasting techniques.
    Timelytics uses a powerful ensemble model to accurately forecast Order-to-Delivery (OTD) times. 
    By leveraging machine learning algorithms, this tool helps businesses identify bottlenecks and reduce delays.
    """
)

# Load the trained ensemble model
model_path = "voting_model.pkl"
try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please upload the `voting_model.pkl` file.")
    st.stop()

# Input parameters
st.sidebar.header("Input Parameters")

purchase_dow = st.sidebar.number_input(
    "Purchase Day of the Week (0 = Sunday, 6 = Saturday)", min_value=0, max_value=6, step=1, value=3
)
purchase_month = st.sidebar.number_input(
    "Purchase Month (1-12)", min_value=1, max_value=12, step=1, value=1
)
year = st.sidebar.number_input("Purchase Year", min_value=2000, max_value=2100, step=1, value=2018)
product_size_cm3 = st.sidebar.number_input("Product Size (cm³)", min_value=0, step=1, value=10000)
product_weight_g = st.sidebar.number_input("Product Weight (grams)", min_value=0, step=1, value=2000)
geolocation_state_customer = st.sidebar.text_input(
    "Customer State (e.g., SP, RJ)", value="SP"
)
geolocation_state_seller = st.sidebar.text_input(
    "Seller State (e.g., SP, RJ)", value="RJ"
)
distance = st.sidebar.number_input("Distance (km)", min_value=0.0, step=0.1, value=300.0)

# Prediction function
def predict_delivery_time(
    dow, month, year, size, weight, customer_state, seller_state, distance
):
    # Dummy encoding for customer and seller state (adjust based on actual model requirements)
    state_encoding = {
        "SP": 1,
        "RJ": 2,
        "MG": 3,
        "ES": 4,
    }
    customer_state_encoded = state_encoding.get(customer_state.upper(), 0)
    seller_state_encoded = state_encoding.get(seller_state.upper(), 0)

    # Prepare input array
    input_array = np.array([
        [dow, month, year, size, weight, customer_state_encoded, seller_state_encoded, distance]
    ])

    # Predict
    prediction = model.predict(input_array)
    return round(prediction[0], 2)

# Predict button
if st.sidebar.button("Predict Delivery Time"):
    with st.spinner("Predicting delivery time..."):
        result = predict_delivery_time(
            purchase_dow,
            purchase_month,
            year,
            product_size_cm3,
            product_weight_g,
            geolocation_state_customer,
            geolocation_state_seller,
            distance,
        )
    st.success(f"Predicted Delivery Time: {result} days")

# Footer
st.markdown(
    """
    #### About Timelytics
    Timelytics leverages historical data and machine learning to optimize supply chain efficiency.
    For more details, visit the project documentation or contact the developer.
    """
)
