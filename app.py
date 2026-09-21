import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('delivery_delay.sav')

# Define the feature columns based on the original training data
feature_columns = ['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
                   'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
                   'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
                   'Warehouse_Processing_Time']

st.title('Delivery Delay Prediction App')
st.write('Enter the details below to predict if there will be a delivery delay.')

# Create input fields for each feature
with st.form('prediction_form'):
    delivery_distance = st.number_input('Delivery Distance (km)', min_value=0.0, value=19.35)
    traffic_congestion = st.slider('Traffic Congestion (1-5, 5 being highest)', 1, 5, 3)
    weather_condition = st.slider('Weather Condition (1-5, 5 being worst)', 1, 5, 2)
    delivery_slot = st.slider('Delivery Slot (1-3, 1=Early, 2=Mid, 3=Late)', 1, 3, 2)
    driver_experience = st.number_input('Driver Experience (years)', min_value=0, value=5)
    num_stops = st.number_input('Number of Stops', min_value=0, value=5)
    vehicle_age = st.number_input('Vehicle Age (years)', min_value=0, value=5)
    road_condition_score = st.slider('Road Condition Score (1-5, 5 being best)', 1, 5, 3)
    package_weight = st.number_input('Package Weight (kg)', min_value=0.0, value=10.0)
    fuel_efficiency = st.number_input('Fuel Efficiency (km/L)', min_value=0.0, value=15.0)
    warehouse_processing_time = st.number_input('Warehouse Processing Time (minutes)', min_value=0, value=60)
   
    submitted = st.form_submit_button('Predict Delivery Delay')

    if submitted:
        # Create a DataFrame from the input values
        input_data = pd.DataFrame([{
            'Delivery_Distance': delivery_distance,
            'Traffic_Congestion': traffic_congestion,
            'Weather_Condition': weather_condition,
            'Delivery_Slot': delivery_slot,
            'Driver_Experience': driver_experience,
            'Num_Stops': num_stops,
            'Vehicle_Age': vehicle_age,
            'Road_Condition_Score': road_condition_score,
            'Package_Weight': package_weight,
            'Fuel_Efficiency': fuel_efficiency,
            'Warehouse_Processing_Time': warehouse_processing_time
        }])

        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[0]

        st.subheader('Prediction Result:')
        if prediction[0] == 1:
            st.error(f'The model predicts a **Delivery Delay** (Probability: {prediction_proba[1]:.2f})')
        else:
            st.success(f'The model predicts **No Delivery Delay** (Probability: {prediction_proba[0]:.2f})')
