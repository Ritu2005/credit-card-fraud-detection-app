import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.title('Credit Card Fraud Detection App')
st.write('Upload your dataset to detect fraudulent transactions using a Random Forest Classifier.')

uploaded_file = st.file_uploader('Upload CSV', type='csv')

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write('### Data Overview')
    st.write(data.head())
    st.write(f'Data Shape: {data.shape}')
    st.write(data.describe())

    fraud = data[data['Class'] == 1]
    valid = data[data['Class'] == 0]
    outlierFraction = len(fraud) / float(len(valid))

    st.write('### Class Distribution')
    st.write(f'Outlier Fraction: {outlierFraction}')
    st.write(f'Fraud Cases: {len(fraud)}')
    st.write(f'Valid Transactions: {len(valid)}')

    fig, ax = plt.subplots()
    sns.countplot(x='Class', data=data, ax=ax)
    st.pyplot(fig)

    # Separate Features and Labels
    X = data.drop(['Class'], axis=1)
    Y = data['Class']
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Build Random Forest Classifier
    st.write('### Training Model')
    rfc = RandomForestClassifier()
    rfc.fit(xTrain, yTrain)
    yPred = rfc.predict(xTest)

    # Evaluation Metrics
    st.write('### Model Evaluation')
    st.write(f'Accuracy: {accuracy_score(yTest, yPred):.4f}')
    st.write(f'Precision: {precision_score(yTest, yPred):.4f}')
    st.write(f'Recall: {recall_score(yTest, yPred):.4f}')
    st.write(f'F1-Score: {f1_score(yTest, yPred):.4f}')

    # Confusion Matrix
    st.write('### Confusion Matrix')
    conf_matrix = confusion_matrix(yTest, yPred)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'], ax=ax)
    st.pyplot(fig)

    st.write('Thank you for using the app! 😊')
else:
    st.write('Please upload a CSV file to proceed.')
