import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load and cache data
@st.cache_data
def load_data():
    # Load local dataset (replace with the actual path)
    df = pd.read_csv(r'C:\Users\GRANTIK THINGALAYA\PycharmProjects\CODSOFT_ML_03\Churn_Modelling.csv')
    return df

# Preprocess data
def preprocess_data(df):
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)  # Drop unnecessary columns
    df = pd.get_dummies(df, drop_first=True)  # Convert categorical variables to dummy variables
    X = df.drop('Exited', axis=1)  # Features
    y = df['Exited']  # Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# Model training and prediction
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

# Streamlit interface
st.title('Customer Churn Prediction')

df = load_data()
st.write('Dataset Preview:')
st.write(df.head())

X_train, X_test, y_train, y_test = preprocess_data(df)

model_option = st.selectbox('Select Model', ['Logistic Regression', 'Random Forest', 'Gradient Boosting'])

if model_option == 'Logistic Regression':
    model = LogisticRegression(max_iter=1000)
elif model_option == 'Random Forest':
    model = RandomForestClassifier()
else:
    model = GradientBoostingClassifier()

st.write(f'Training {model_option}...')
accuracy, report = train_and_evaluate(model, X_train, X_test, y_train, y_test)

st.write(f'Accuracy: {accuracy:.2f}')
st.write('Classification Report:')
st.text(report)
