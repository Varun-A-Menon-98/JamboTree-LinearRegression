import streamlit as st
import pandas as pd
import numpy as np
import os
import yaml  # Importing the yaml module
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
import logging

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
        input_array = np.array(input_values).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)
        return prediction[0]
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        st.error("There was an error during prediction.")
        raise e

# Streamlit UI and Interaction
def run_streamlit_app():
    # Load configuration to get the file path
    # use ehen using local
    # config = load_config()
    # file_path = config.get('PATH')

    # # Make sure the file path exists
    # if not file_path or not os.path.exists(file_path):
    #     st.error(f"File path '{file_path}' is invalid or missing.")
    #     return

    # use when URL
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

    # Create sliders for each feature dynamically
    for feature in feature_names:
        min_value = float(data[feature].min())
        max_value = float(data[feature].max())
        step = (max_value - min_value) / 10  # You can adjust the step for better control
        value = st.slider(feature, min_value, max_value, (min_value + max_value) / 2, step)
        input_values.append(value)

    # Display the prediction when the user presses the button
    if st.button("Predict Chance of Admit"):
        prediction = predict_admission(input_values, model, scaler, feature_names)
        st.write(f"The predicted Chance of Admit is {np.clip(prediction * 100, 0, 100):.3f}%")

# Corrected the __name__ check
if __name__ == "__main__":
    run_streamlit_app()
