import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Set up Streamlit page configuration
st.set_page_config(page_title="Money Laundering Detection", page_icon="💸", layout="wide")

# Add FontAwesome for icons
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">', unsafe_allow_html=True)

# Create a sidebar for navigation
st.sidebar.title("Navigation")
options = ["🏠 Home", "🔍 Prediction"]
choice = st.sidebar.selectbox("Go to", options)

# Define a function to upload and read CSV files
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

# Function to preprocess data and train the model
def process_and_train(data):
    X = data[['Amount Received', 'Amount Paid', 'Payment Format']]
    y = data['Is Laundering']
    
    # Encode categorical columns
    label_encoders = {}
    for col in ['Payment Format']:
        label_encoders[col] = LabelEncoder()
        X[col] = label_encoders[col].fit_transform(X[col])
    
    # Encode target variable
    y_label_encoder = LabelEncoder()
    y = y_label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model and encoders to a pickle file
    with open("random_forest_model.pkl", "wb") as f:
        pickle.dump((model, label_encoders, y_label_encoder), f)
    
    return model, label_encoders, y_label_encoder

# Initialize uploaded_file variable
uploaded_file = st.sidebar.file_uploader("Upload your CSV file here", type=["csv"])

# Variables to store data and processed models
data = None
model = None
label_encoders = None
y_label_encoder = None

# Home Page
if choice == "🏠 Home":
    st.title("Money Laundering Detection Application")
    st.write("Drag and drop your CSV files below to start.")
    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        data = load_data(uploaded_file)
        st.image("https://via.placeholder.com/800x300?text=Money+Laundering+Detection", use_column_width=True)

# Prediction
if choice == "🔍 Prediction":
    st.write("### Upload File for Batch Prediction")
    
    uploaded_batch_file = st.file_uploader("Upload a CSV file for batch prediction", type=["csv"])
    if uploaded_batch_file is not None:
        # Load model from pickle file if not already loaded
        if model is None:
            with open("random_forest_model.pkl", "rb") as f:
                model, label_encoders, y_label_encoder = pickle.load(f)
        
        batch_data = pd.read_csv(uploaded_batch_file)
        if all(col in batch_data.columns for col in ['Amount Received', 'Amount Paid', 'Payment Format']):
            # Encode categorical features
            batch_data['Payment Format'] = label_encoders['Payment Format'].transform(batch_data['Payment Format'])
            
            # Make predictions on batch data
            batch_predictions = model.predict(batch_data[['Amount Received', 'Amount Paid', 'Payment Format']])
            
            # Calculate percentages
            laundering_percentage = np.mean(batch_predictions == 1) * 100
            not_laundering_percentage = np.mean(batch_predictions == 0) * 100

            st.write(f'Percentage of transactions classified as Money Laundering: {laundering_percentage:.2f}%')
            st.write(f'Percentage of transactions classified as Not Laundering: {not_laundering_percentage:.2f}%')

            # Display batch predictions
            batch_data['Prediction'] = y_label_encoder.inverse_transform(batch_predictions)
            st.write("### Batch Predictions")
            st.dataframe(batch_data[['Amount Received', 'Amount Paid', 'Payment Format', 'Prediction']])
        else:
            st.error("The uploaded file must contain the columns: 'Amount Received', 'Amount Paid', 'Payment Format'.")
    else:
        st.write("Please upload a CSV file to perform batch predictions.")
