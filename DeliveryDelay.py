import joblib
import pandas as pd

# Define the path to your saved model
MODEL_PATH = 'delivery_delay.sav'

def load_model(model_path):
    """Loads the pre-trained Logistic Regression model."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None

def predict_delivery_delay(model, input_data):
    """Makes a prediction using the loaded model."

    Args:
        model: The trained Logistic Regression model.
        input_data (list or array-like): A single sample of feature values
                                          matching the training data.

    Returns:
        int: The predicted delivery delay (0 for no delay, 1 for delay).
    """
    # Convert input_data to a DataFrame with appropriate column names
    # This ensures the model receives data in the expected format.
    # You need to ensure 'X.columns' is available or manually define column names.
    # For this example, we'll assume the feature names are known.

    # IMPORTANT: Replace these with your actual feature names if they are different
    feature_names = [
        'Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition', 
        'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age', 
        'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency', 
        'Warehouse_Processing_Time'
    ]

    if not isinstance(input_data, pd.DataFrame):
        input_df = pd.DataFrame([input_data], columns=feature_names)
    else:
        input_df = input_data
        
    prediction = model.predict(input_df)
    return prediction[0]


if __name__ == "__main__":
    # Load the model
    logi_model = load_model(MODEL_PATH)

    if logi_model:
        print("Model loaded successfully!")

        # Example usage: Make a prediction with new data
        # This should be a single row of features, matching the order
        # and type of features used during training.
        sample_new_data = [
            12, 3, 1, 2, 4, 2, 8, 2, 15, 12, 80
        ]  # Example values

        predicted_delay = predict_delivery_delay(logi_model, sample_new_data)

        if predicted_delay == 1:
            print(f"Predicted Delivery Delay: Yes (1)")
        else:
            print(f"Predicted Delivery Delay: No (0)")

        # You can add more complex input handling or API integration here
    else:
        print("Could not load the model. Please ensure 'delivery_delay.sav' is in the correct directory.")
