import streamlit as st
import joblib
import pandas as pd

# Define the path to your saved model
MODEL_PATH = 'delivery_delay.sav'

# Feature names - IMPORTANT: These must match the order and names used during training
feature_names = [
    'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition', 
    'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age', 
    'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency', 
    'Warehouse_Processing_Time'
]

# Load the model
@st.cache_resource
def load_model(path):
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {path}. Please ensure 'delivery_delay.sav' is in the same directory as the Streamlit app.")
        return None

logi_model = load_model(MODEL_PATH)

st.title('Delivery Delay Prediction App')
st.write('Enter the details below to predict if there will be a delivery delay.')

if logi_model:
    # Create input widgets for each feature
    input_features = {}
    col1, col2 = st.columns(2)

    with col1:
        input_features['Delivery_Distance'] = st.number_input('Delivery Distance', min_value=0.0, value=20.0, help='Distance of the delivery in km.')
        input_features['Traffic_Congestion'] = st.slider('Traffic Congestion', min_value=1, max_value=5, value=3, help='Level of traffic congestion (1=low, 5=high).')
        input_features['Weather_Condition'] = st.slider('Weather Condition', min_value=1, max_value=3, value=2, help='Weather conditions (1=good, 2=moderate, 3=bad).')
        input_features['Delivery_Slot'] = st.slider('Delivery Slot', min_value=1, max_value=3, value=2, help='Preferred delivery time slot (1=morning, 2=afternoon, 3=evening).')
        input_features['Driver_Experience'] = st.number_input('Driver Experience (years)', min_value=0, value=5, help='Years of experience of the driver.')
        input_features['Num_Stops'] = st.number_input('Number of Stops', min_value=1, value=5, help='Total number of stops on the delivery route.')

    with col2:
        input_features['Vehicle_Age'] = st.number_input('Vehicle Age (years)', min_value=0, value=3, help='Age of the delivery vehicle.')
        input_features['Road_Condition_Score'] = st.slider('Road Condition Score', min_value=1, max_value=4, value=2, help='Rating of road conditions (1=poor, 4=excellent).')
        input_features['Package_Weight'] = st.number_input('Package Weight (kg)', min_value=0.0, value=10.0, help='Weight of the package in kg.')
        input_features['Fuel_Efficiency'] = st.number_input('Fuel Efficiency (km/l)', min_value=0.0, value=15.0, help='Fuel efficiency of the vehicle in km/liter.')
        input_features['Warehouse_Processing_Time'] = st.number_input('Warehouse Processing Time (minutes)', min_value=0, value=60, help='Time taken for package processing at the warehouse in minutes.')

    # Convert input to DataFrame for prediction
    input_df = pd.DataFrame([input_features])

    # Make prediction
    if st.button('Predict Delivery Delay'):
        prediction = logi_model.predict(input_df)
        prediction_proba = logi_model.predict_proba(input_df)

        st.subheader('Prediction Result:')
        if prediction[0] == 1:
            st.error(f"Predicted: **Delivery Delay is LIKELY**")
            st.write(f"Probability of Delay: {prediction_proba[0][1]:.2f}")
            st.write(f"Probability of No Delay: {prediction_proba[0][0]:.2f}")
        else:
            st.success(f"Predicted: **No Delivery Delay**")
            st.write(f"Probability of No Delay: {prediction_proba[0][0]:.2f}")
            st.write(f"Probability of Delay: {prediction_proba[0][1]:.2f}")

    st.markdown("""
    --- 
    **How to run this Streamlit app:**
    1. Save the code above as `streamlit_app.py` in the same directory as your `delivery_delay.sav` model file.
    2. Open your terminal or command prompt.
    3. Navigate to that directory.
    4. Run the command: `streamlit run streamlit_app.py`
    5. The app will open in your web browser.
    """)
else:
    st.warning("Model could not be loaded. Please check the model file path.")
