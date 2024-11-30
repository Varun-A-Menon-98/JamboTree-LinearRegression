import streamlit as st
import pandas as pd
import numpy as np
import os
import yaml  # Importing the yaml module
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
import logging
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Load configuration settings
def load_config():
    try:
        with open('config.yaml', 'r') as config_file:  # Read config.yaml file
            config = yaml.safe_load(config_file)
            return config
    except FileNotFoundError:
        logger.error("Configuration file not found.")
        st.error("Configuration file not found.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error reading the YAML file: {e}")
        st.error("Error reading the YAML file.")
        raise

# Load the dataset
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        data.rename(columns={'Chance of Admit ': 'Chance of Admit'}, inplace=True)
        data.drop(columns="Serial No.", inplace=True)
        return data
    except FileNotFoundError:
        logger.error(f"File {file_path} not found.")
        st.error(f"File {file_path} not found.")
        raise
    except pd.errors.EmptyDataError:
        logger.error("The dataset is empty.")
        st.error("The dataset is empty.")
        raise

# Preprocess the dataset and train the selected model (Ridge or Lasso)
def preprocess_and_train(data, model_type="Ridge"):
    # Prepare features and target
    X = data.drop(columns=["Chance of Admit", "SOP", "University Rating"])
    y = data['Chance of Admit']

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Select and train the model based on the choice
    if model_type == "Ridge":
        model = Ridge(alpha=10)  # You can adjust the alpha value for Ridge
    elif model_type == "Lasso":
        model = Lasso(alpha=0.01)  # Lasso with alpha=0.01
    else:
        raise ValueError("Unsupported model type")

    model.fit(X_scaled, y)

    return model, scaler, X.columns.tolist()

# Function to make predictions
def predict_admission(input_values, model, scaler, feature_names):
    try:
        # Reshaping input values and scaling them for prediction
        input_array = np.array(input_values).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)
        return prediction[0]  # Return the first prediction value (since it's a single value)
    except Exception as e:
        # Handle any errors that occur during the prediction
        logger.error(f"Error during prediction: {str(e)}")
        st.error("There was an error during prediction.")
        raise e  # Re-raise the error after logging
 # This will re-raise the error after logging

# Streamlit UI and Interaction
def run_streamlit_app():
    # Load configuration to get the file path
    config = load_config()
    file_path = config.get('PATH')

    # Check if the file path is a valid URL
    if not file_path:
        st.error("File path is missing in the configuration.")
        return

    try:
        # Check if the file URL is accessible
        response = requests.get(file_path)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to access the file at {file_path}. Error: {e}")
        return

    # Load the data
    data = load_data(file_path)

    # Streamlit UI elements
    st.title("University Admission Prediction")

    # Dropdown to select model type (Ridge or Lasso)
    model_type = st.selectbox("Select the Model Type", ("Ridge", "Lasso"))

    # Preprocess data and train the selected model
    model, scaler, feature_names = preprocess_and_train(data, model_type=model_type)

    input_values = []

    # Create input boxes for each feature dynamically
    for feature in feature_names:
        if feature == "Research":  # Special handling for "Research" feature (toggle)
            value = st.toggle(feature)
            input_values.append(value)  # Add either 0 or 1 based on the toggle
        else:
            try:
                # Ensure the feature is numeric before using it

        # Check if the feature is numeric
                if pd.api.types.is_numeric_dtype(data[feature]):
                    max_value = float(data[feature].max())  # Get the max value of the feature
                    min_value = 0  # Start from 0
        
                    # Handle CGPA with a step of 0.1
                    if feature == 'CGPA':
                        value = st.number_input(
                            feature, min_value=min_value, max_value=max_value, 
                            value=max_value // 2, step=0.1)  # Step size of 0.1 for CGPA
                    else:
                        # Handle other numeric features with a step size of 1
                        value = st.number_input(
                            feature, min_value=min_value, max_value=int(max_value), 
                            value=int(max_value) // 2, step=1)  # Step size of 1 for other numeric features
                    
                    input_values.append(value)
                else:
                # For non-numeric features, you can handle them here or ignore
                    st.warning(f"Feature '{feature}' is not numeric. Skipping input box.")
                    input_values.append(0)  # Add a default value for non-numeric features
            except Exception as e:
            # Catch errors during the creation of the input box
                st.warning(f"Error creating input box for '{feature}': {e}")
                input_values.append(0) 
            #Add a default value in case of error

    # Display the input values for debugging purposes
    # st.write("Input values:", input_values)

    # Ensure the input has the correct shape for prediction (2D array)
    input_array = np.array(input_values).reshape(1, -1)

    # Predict when the user presses the button
    if st.button("Predict Chance of Admit"):
        try:
            prediction = predict_admission(input_array, model, scaler, feature_names)
            prediction_percent = np.clip(prediction * 100, 0, 100)
            
            # Display larger font and center text
            st.markdown(f"<h2 style='text-align: center; font-size: 30px;'>The predicted Chance of Admit is {prediction_percent:.3f}%</h2>", unsafe_allow_html=True)
            
            # If prediction is greater than 90, show balloons
            if prediction_percent > 90:
                st.balloons()

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")





# Corrected the __name__ check
if __name__ == "__main__":
    run_streamlit_app()
