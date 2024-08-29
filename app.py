# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb

# Set up Streamlit page configuration
st.set_page_config(page_title="Money Laundering Detection", page_icon="💸", layout="wide")

# Create a sidebar for navigation
st.sidebar.title("Navigation")
options = ["Home", "Fetch Data", "Data Filtering", "Data Slicing", "Model Generation", "Prediction"]
choice = st.sidebar.selectbox("Go to", options)

# Define a function to upload and read CSV files
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    return data

# Initialize uploaded_file variable
uploaded_file = st.sidebar.file_uploader("Upload your CSV file here", type=["csv"])

# Home Page
if choice == "Home":
    st.title("Money Laundering Detection Application")
    st.write("Drag and drop your CSV files below to start.")
    if uploaded_file is not None:
        st.success("File uploaded successfully!")

# Fetch Data
if choice == "Fetch Data":
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("### Dataset Preview")
        st.dataframe(data.head())
        st.write("### Column Names")
        st.write(data.columns)
    else:
        st.write("Please upload a CSV file to fetch data.")


# Data Filtering
if choice == "Data Filtering":
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("### Check for Null Values")
        st.write(data.isnull().sum())
    else:
        st.write("Please upload a CSV file to filter data.")

# Data Slicing
if choice == "Data Slicing":
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("### Categorical Variable Counts")
        st.write(data['Is Laundering'].value_counts())
    else:
        st.write("Please upload a CSV file to slice data.")


if choice == "Model Generation":
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("### Model Training")
        
        # Preprocess data
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
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Save model and encoders to a pickle file
        with open("random_forest_model.pkl", "wb") as f:
            pickle.dump((rf, label_encoders, y_label_encoder), f)

        # Bar plot for Payment Format
        st.write("### Payment Format Usage in Money Laundering")
        plt.figure(figsize=(8, 6))
        sns.countplot(data=data, x='Payment Format', hue='Is Laundering')
        plt.title('Payment Format vs. Money Laundering')
        plt.xlabel('Payment Format')
        plt.ylabel('Count')
        st.pyplot(plt)
        
        # Count plot for Is Laundering
        st.write("### Distribution of Money Laundering")
        plt.figure(figsize=(8, 6))
        sns.countplot(data=data, x='Is Laundering')
        plt.title('Distribution of Money Laundering')
        plt.xlabel('Is Laundering')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['Not Laundering', 'Laundering'])
        st.pyplot(plt)

    else:
        st.write("Please upload a CSV file to generate a model.")
        

# Prediction
if choice == "Prediction":
    if uploaded_file is not None:
        st.write("### Enter Transaction Details")
        
        from_bank_id = st.text_input("From Bank ID")
        account_no = st.text_input("Account No")
        to_bank_id = st.text_input("To Bank ID")
        account_no1 = st.text_input("Account No1")
        amount_received = st.number_input("Amount Received", min_value=0.0)
        receiving_currency = st.selectbox("Receiving Currency", ['USD', 'Euro', 'Yuan', 'Yen', 'Rupee', 'Brazil Real', 'Mexican Peso', 'Shekel', 'Bitcoin', 'Ruble', 'Swiss Franc', 'UK Pound', 'Australian Dollar', 'Saudi Riyal', 'Canadian Dollar'])
        amount_paid = st.number_input("Amount Paid", min_value=0.0)
        payment_currency = st.selectbox("Payment Currency", ['USD', 'Euro', 'Yuan', 'Yen', 'Rupee', 'Brazil Real', 'Mexican Peso', 'Shekel', 'Bitcoin', 'Ruble', 'Swiss Franc', 'UK Pound', 'Australian Dollar', 'Saudi Riyal', 'Canadian Dollar'])
        payment_format = st.selectbox("Payment Format", ['Online', 'Wire', 'Cash', 'Cheque', 'ACH', 'Bitcoin', 'Credit Card', 'Reinvestment'])

        
        if st.button("Predict"):
            # Load model from pickle file
            with open("random_forest_model.pkl", "rb") as f:
                model, label_encoders, y_label_encoder = pickle.load(f)
            
            # Preprocess the input data
            input_data = pd.DataFrame({
                'Amount Received': [amount_received],
                'Amount Paid': [amount_paid],
                'Payment Format': [payment_format],
            })
            
            # Encode categorical features
            input_data['Payment Format'] = label_encoders['Payment Format'].transform(input_data['Payment Format'])
            
            # Make prediction
            prediction = model.predict(input_data)
            if prediction[0] == 1:
                st.write("### Prediction: This transaction is suspicious for money laundering.")
            else:
                st.write("### Prediction: This transaction is not suspicious for money laundering.")
    else:
        st.write("Please upload a CSV file to make predictions.")
        
# # Performance Estimation
# if choice == "Performance Estimation":
#     if uploaded_file is not None:
#         data = load_data(uploaded_file)
#         st.write("### Model Performance")
        
#         # Re-split the data
#         X = data[['Amount Received', 'Amount Paid', 'Payment Format']]
#         y = data['Is Laundering']
        
#         # Encode categorical columns
#         label_encoders = {}
#         for col in ['Payment Format']:
#             label_encoders[col] = LabelEncoder()
#             X[col] = label_encoders[col].fit_transform(X[col])
        
#         # Encode target variable
#         y_label_encoder = LabelEncoder()
#         y = y_label_encoder.fit_transform(y)

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
#         # Load the model
#         with open("random_forest_model.pkl", "rb") as f:
#             rf, saved_label_encoders, saved_y_label_encoder = pickle.load(f)
        
#         # Ensure the same label encoding is applied to the test data
#         X_test['Payment Format'] = saved_label_encoders['Payment Format'].transform(X_test['Payment Format'])
        
#         y_test = saved_y_label_encoder.transform(y_test)
        
#         y_pred = rf.predict(X_test)
        
#         # Display model performance
#         accuracy = accuracy_score(y_test, y_pred)
#         conf_matrix = confusion_matrix(y_test, y_pred)
#         class_report = classification_report(y_test, y_pred)

#         st.write(f'Accuracy: {accuracy}')
#         st.write('Confusion Matrix:')
#         st.write(conf_matrix)
#         st.write('Classification Report:')
#         st.write(class_report)
#     else:
#         st.write("Please upload a CSV file to estimate model performance.")
