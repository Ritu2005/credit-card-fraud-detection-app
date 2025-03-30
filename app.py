import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.title('🚀 Credit Card Fraud Detection App')
st.write('Upload your dataset to detect fraudulent transactions using a Random Forest Classifier.')

uploaded_file = st.file_uploader('📂 Upload CSV', type='csv')

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Check for required columns
    if 'Class' not in data.columns:
        st.error('The uploaded CSV must contain a "Class" column.')
    else:
        st.write('### 📊 Data Overview')
        st.write(data.head())
        st.write(f'Data Shape: {data.shape}')
        st.write(data.describe())

        # Class distribution
        fraud = data[data['Class'] == 1]
        valid = data[data['Class'] == 0]
        outlierFraction = len(fraud) / float(len(valid))
        st.write(f'Outlier Fraction: {outlierFraction:.4f}')
        st.write(f'Fraud Cases: {len(fraud)}')
        st.write(f'Valid Transactions: {len(valid)}')

        fig, ax = plt.subplots()
        sns.countplot(x='Class', data=data, ax=ax)
        st.pyplot(fig)

        # Features and labels
        X = data.drop(['Class'], axis=1)
        Y = data['Class']
        xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Hyperparameter selection
        n_estimators = st.slider('Number of Trees (n_estimators)', 50, 500, 100, step=50)
        max_depth = st.slider('Max Depth of Trees (max_depth)', 2, 20, 10)

        # Train model
        st.write('### 🛠️ Training Model')
        rfc = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        rfc.fit(xTrain, yTrain)
        yPred = rfc.predict(xTest)

        # Evaluation metrics
        st.write('### 📈 Model Evaluation')
        st.write(f'Accuracy: {accuracy_score(yTest, yPred):.4f}')
        st.write(f'Precision: {precision_score(yTest, yPred):.4f}')
        st.write(f'Recall: {recall_score(yTest, yPred):.4f}')
        st.write(f'F1-Score: {f1_score(yTest, yPred):.4f}')

        # Confusion matrix
        st.write('### 📊 Confusion Matrix')
        conf_matrix = confusion_matrix(yTest, yPred)
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'], ax=ax)
        st.pyplot(fig)

        # Feature Importance
        st.write('### 🔎 Feature Importance')
        feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rfc.feature_importances_})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
        fig, ax = plt.subplots()
        sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
        st.pyplot(fig)

        # Prediction on Custom Data
        st.write('### 🤖 Predict Fraud on New Transaction')
        user_input = []
        for col in X.columns:
            user_input.append(st.number_input(f'{col}', value=float(data[col].mean())))

        if st.button('Predict'): 
            prediction = rfc.predict([user_input])
            st.write('Prediction:', '🟢 Not Fraud' if prediction[0] == 0 else '🔴 Fraud')

else:
    st.write('Please upload a CSV file to proceed.')
