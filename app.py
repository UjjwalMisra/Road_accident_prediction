import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import numpy as np

# Load the dataset (DATA COLLECTION)
@st.cache_data
def load_data():
    data = pd.read_csv(r"C:\Users\KIIT\OneDrive\Desktop\miniproject\dataset.csv.csv")
    return data

data = load_data()

# Display basic info (PREPROCESSING: Data Understanding)
st.write("Dataset Shape:", data.shape)
st.write("Class Distribution:")
st.write(data['Accident'].value_counts())

# Handle missing values - first fill with forward fill, then with mode for categorical and median for numerical
for col in data.columns:
    if data[col].dtype == 'object':
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        data[col].fillna(data[col].median(), inplace=True)

# Create manual encoders for categorical variables to ensure consistency
categorical_cols = ['Weather', 'Road_Type', 'Time_of_Day', 'Traffic_Density',
                   'Accident_Severity', 'Road_Condition', 'Vehicle_Type',
                   'Road_Light_Condition']

encoders = {}
for col in categorical_cols:
    if col in data.columns:
        unique_values = data[col].dropna().unique()
        encoders[col] = {val: idx for idx, val in enumerate(unique_values)}
        data[col] = data[col].map(encoders[col])

# Split the data
X = data.drop('Accident', axis=1)
y = data['Accident']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create a pipeline with imputation, scaling, and model
model = make_pipeline(
    SimpleImputer(strategy='median'),  # Handles any remaining missing values
    StandardScaler(),
    LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
)

model.fit(X_train, y_train)

# Streamlit UI
st.title("Traffic Accident Prediction")

# Define options for categorical features
input_options = {
    'Weather': ['Clear', 'Rainy', 'Foggy', 'Stormy'],
    'Road_Type': ['City Road', 'Rural Road', 'Highway', 'Mountain Road'],
    'Time_of_Day': ['Morning', 'Afternoon', 'Evening', 'Night'],
    'Traffic_Density': ['0', '1', '2'],
    'Driver_Alcohol': ['0', '1'],
    'Driver_Age': ['18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', 
 '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', 
 '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', 
 '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', 
 '67', '68', '69', '70'],
    'Speed_Limit': ['20', '30', '40', '50', '60', '70', '80', '90', '100', '110', '120', '130', '140', '150', '160', '170', '180', '190', '200'],
    'Accident_Severity': ['Low', 'Moderate', 'High'],
    'Number_of_Vehicles': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'Road_Condition': ['Dry', 'Wet', 'Icy', 'Under Construction'],
    'Vehicle_Type': ['Car', 'Truck', 'Bus', 'Motorcycle'],
    'Road_Light_Condition': ['Daylight', 'Artificial Light']
}

# Create input form
user_input = {}
for col in X.columns:
    if col in input_options:
        user_input[col] = st.selectbox(col, input_options[col])
    else:
        # For numerical columns
        min_val = float(data[col].min())
        max_val = float(data[col].max())
        default_val = float(data[col].median())
        user_input[col] = st.number_input(col, min_value=min_val, max_value=max_val, value=default_val)

if st.button("Predict"):
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([user_input])
        
        # Encode categorical variables
        for col in categorical_cols:
            if col in input_df.columns:
                input_df[col] = input_df[col].map(encoders.get(col, {}))
        
        # Ensure all columns are numeric and in correct order
        input_df = input_df[X.columns].apply(pd.to_numeric)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        # Display results
        st.subheader("Prediction Results")
        if prediction == 1:
            st.error(f"🚨 Accident Likely (Probability: {probability:.2%})")
            st.write("Warning: High risk of accident with these conditions")
        else:
            st.success(f"✅ No Accident Likely (Probability: {probability:.2%})")
            st.write("Conditions appear safe for travel")
        
        # Show feature importance
        st.subheader("Key Factors Influencing This Prediction")
        if hasattr(model.named_steps['logisticregression'], 'coef_'):
            coefs = pd.Series(model.named_steps['logisticregression'].coef_[0], index=X.columns)
            top_factors = coefs.abs().sort_values(ascending=False).head(5)
            for factor in top_factors.index:
                st.write(f"- {factor}: {coefs[factor]:.2f}")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your input values and try again.")