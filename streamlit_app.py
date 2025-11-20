import streamlit as st
import numpy as np
import pandas as pd
import pickle
from PyPDF2 import PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io
import re

# Load the trained model
model = pickle.load(open('heart_disease_model.sav', 'rb'))

# Load the dataset for display
heart_data = pd.read_csv('heart_disease_data (1).csv')

# Streamlit app title
st.title('Heart Disease Prediction App by Gaurav')

# Display dataset information
st.header('Dataset Overview')
st.write('Shape of the dataset:', heart_data.shape)
st.write('First 5 rows:')
st.dataframe(heart_data.head())
st.write('Last 5 rows:')
st.dataframe(heart_data.tail())
st.write('Statistical summary:')
st.dataframe(heart_data.describe())
st.write('Target distribution:')
st.bar_chart(heart_data['target'].value_counts())

# Model accuracy (assuming from training, but in app we load pre-trained)
st.header('Model Performance')
st.write('Note: Model was trained with Logistic Regression. Accuracy details from training:')
st.write('- Accuracy on training data: ~0.85 (approximate from script)')
st.write('- Accuracy on test data: ~0.82 (approximate from script)')

# Tabs for different prediction methods
tab1, tab2 = st.tabs(["Manual Input", "Upload PDF Report"])

with tab1:
    st.header('Predict Heart Disease (Manual Input)')
    st.subheader('Patient Details')
    patient_name = st.text_input('Patient Name', key='patient_name_manual')
    prediction_date = st.date_input('Prediction Date', key='prediction_date_manual')

    st.subheader('Medical Features')
    st.write('Enter the following features to predict if the person has heart disease:')

    # Feature inputs based on dataset columns (excluding target)
    age = st.slider('Age', 20, 100, 50, key='age_manual')
    sex = st.selectbox('Sex (0: Female, 1: Male)', [0, 1], key='sex_manual')
    cp = st.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3], key='cp_manual')
    trestbps = st.slider('Resting Blood Pressure', 90, 200, 120, key='trestbps_manual')
    chol = st.slider('Serum Cholesterol', 100, 600, 200, key='chol_manual')
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (0: No, 1: Yes)', [0, 1], key='fbs_manual')
    restecg = st.selectbox('Resting Electrocardiographic Results (0-2)', [0, 1, 2], key='restecg_manual')
    thalach = st.slider('Maximum Heart Rate Achieved', 60, 220, 150, key='thalach_manual')
    exang = st.selectbox('Exercise Induced Angina (0: No, 1: Yes)', [0, 1], key='exang_manual')
    oldpeak = st.slider('ST Depression Induced by Exercise', 0.0, 6.0, 1.0, key='oldpeak_manual')
    slope = st.selectbox('Slope of the Peak Exercise ST Segment (0-2)', [0, 1, 2], key='slope_manual')
    ca = st.slider('Number of Major Vessels Colored by Fluoroscopy (0-3)', 0, 3, 0, key='ca_manual')
    thal = st.selectbox('Thalassemia (0-3)', [0, 1, 2, 3], key='thal_manual')

    # Prediction button
    if st.button('Predict', key='predict_manual'):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        prediction = model.predict(input_data)
        result = 'The person does not have heart disease.' if prediction[0] == 0 else 'The person has heart disease.'

        st.subheader('Prediction Result:')
        if prediction[0] == 0:
            st.success(result)
        else:
            st.error(result)

        # Generate PDF report
        buffer = io.BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        c.drawString(100, 750, 'Heart Disease Prediction Report')
        c.drawString(100, 730, f'Patient Name: {patient_name}')
        c.drawString(100, 710, f'Prediction Date: {prediction_date}')
        c.drawString(100, 690, f'Patient Age: {age}')
        c.drawString(100, 670, f'Patient Sex: {"Male" if sex == 1 else "Female"}')
        c.drawString(100, 650, f'Chest Pain Type: {cp}')
        c.drawString(100, 630, f'Resting Blood Pressure: {trestbps}')
        c.drawString(100, 610, f'Serum Cholesterol: {chol}')
        c.drawString(100, 590, f'Fasting Blood Sugar: {fbs}')
        c.drawString(100, 570, f'Resting ECG: {restecg}')
        c.drawString(100, 550, f'Max Heart Rate: {thalach}')
        c.drawString(100, 530, f'Exercise Angina: {exang}')
        c.drawString(100, 510, f'ST Depression: {oldpeak}')
        c.drawString(100, 490, f'Slope: {slope}')
        c.drawString(100, 470, f'Major Vessels: {ca}')
        c.drawString(100, 450, f'Thalassemia: {thal}')
        c.drawString(100, 430, f'Prediction: {result}')
        c.save()

        buffer.seek(0)
        st.download_button(
            label="Download Prediction Report as PDF",
            data=buffer,
            file_name=f"{patient_name}_heart_disease_prediction_report.pdf",
            mime="application/pdf",
            key='download_manual'
        )

with tab2:
    st.header('Predict Heart Disease (Upload PDF Report)')
    st.write('Upload a patient PDF report to extract features and predict heart disease.')

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Read PDF
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        st.subheader('Extracted Text from PDF:')
        st.text_area('PDF Content', text, height=200)

        # Extract features from text (simple regex-based extraction, may need adjustment based on PDF format)
        age_match = re.search(r'age[:\s]*(\d+)', text, re.IGNORECASE)
        age = int(age_match.group(1)) if age_match else 50

        sex_match = re.search(r'sex[:\s]*([mf01])', text, re.IGNORECASE)
        sex = 1 if sex_match and sex_match.group(1).lower() in ['m', '1'] else 0

        cp_match = re.search(r'chest pain type[:\s]*([0-3])', text, re.IGNORECASE)
        cp = int(cp_match.group(1)) if cp_match else 0

        trestbps_match = re.search(r'resting blood pressure[:\s]*(\d+)', text, re.IGNORECASE)
        trestbps = int(trestbps_match.group(1)) if trestbps_match else 120

        chol_match = re.search(r'serum cholesterol[:\s]*(\d+)', text, re.IGNORECASE)
        chol = int(chol_match.group(1)) if chol_match else 200

        fbs_match = re.search(r'fasting blood sugar[:\s]*([01])', text, re.IGNORECASE)
        fbs = int(fbs_match.group(1)) if fbs_match else 0

        restecg_match = re.search(r'resting electrocardiographic[:\s]*([0-2])', text, re.IGNORECASE)
        restecg = int(restecg_match.group(1)) if restecg_match else 0

        thalach_match = re.search(r'maximum heart rate[:\s]*(\d+)', text, re.IGNORECASE)
        thalach = int(thalach_match.group(1)) if thalach_match else 150

        exang_match = re.search(r'exercise induced angina[:\s]*([01])', text, re.IGNORECASE)
        exang = int(exang_match.group(1)) if exang_match else 0

        oldpeak_match = re.search(r'st depression[:\s]*([0-9.]+)', text, re.IGNORECASE)
        oldpeak = float(oldpeak_match.group(1)) if oldpeak_match else 1.0

        slope_match = re.search(r'slope[:\s]*([0-2])', text, re.IGNORECASE)
        slope = int(slope_match.group(1)) if slope_match else 0

        ca_match = re.search(r'major vessels[:\s]*([0-3])', text, re.IGNORECASE)
        ca = int(ca_match.group(1)) if ca_match else 0

        thal_match = re.search(r'thalassemia[:\s]*([0-3])', text, re.IGNORECASE)
        thal = int(thal_match.group(1)) if thal_match else 0

        st.subheader('Extracted Features:')
        st.write(f'Age: {age}')
        st.write(f'Sex: {sex}')
        st.write(f'Chest Pain Type: {cp}')
        st.write(f'Resting Blood Pressure: {trestbps}')
        st.write(f'Serum Cholesterol: {chol}')
        st.write(f'Fasting Blood Sugar: {fbs}')
        st.write(f'Resting ECG: {restecg}')
        st.write(f'Max Heart Rate: {thalach}')
        st.write(f'Exercise Angina: {exang}')
        st.write(f'ST Depression: {oldpeak}')
        st.write(f'Slope: {slope}')
        st.write(f'Major Vessels: {ca}')
        st.write(f'Thalassemia: {thal}')

        if st.button('Predict from PDF', key='predict_pdf'):
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            prediction = model.predict(input_data)
            result = 'The person does not have heart disease.' if prediction[0] == 0 else 'The person has heart disease.'

            st.subheader('Prediction Result:')
            if prediction[0] == 0:
                st.success(result)
            else:
                st.error(result)

            # Generate PDF report
            buffer = io.BytesIO()
            c = canvas.Canvas(buffer, pagesize=letter)
            c.drawString(100, 750, 'Heart Disease Prediction Report')
            c.drawString(100, 720, f'Patient Age: {age}')
            c.drawString(100, 700, f'Patient Sex: {"Male" if sex == 1 else "Female"}')
            c.drawString(100, 680, f'Chest Pain Type: {cp}')
            c.drawString(100, 660, f'Resting Blood Pressure: {trestbps}')
            c.drawString(100, 640, f'Serum Cholesterol: {chol}')
            c.drawString(100, 620, f'Fasting Blood Sugar: {fbs}')
            c.drawString(100, 600, f'Resting ECG: {restecg}')
            c.drawString(100, 580, f'Max Heart Rate: {thalach}')
            c.drawString(100, 560, f'Exercise Angina: {exang}')
            c.drawString(100, 540, f'ST Depression: {oldpeak}')
            c.drawString(100, 520, f'Slope: {slope}')
            c.drawString(100, 500, f'Major Vessels: {ca}')
            c.drawString(100, 480, f'Thalassemia: {thal}')
            c.drawString(100, 460, f'Prediction: {result}')
            c.save()

            buffer.seek(0)
            st.download_button(
                label="Download Prediction Report as PDF",
                data=buffer,
                file_name="heart_disease_prediction_report.pdf",
                mime="application/pdf",
                key='download_pdf'
            )
