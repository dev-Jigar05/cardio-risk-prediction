import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="Cardio Risk Predoctor", page_icon="🫀")

API_URL = "https://cardio-risk-prediction-1.onrender.com"

st.title("Cardio Disease Prediction")
st.write("Enter the details below to estimate cardiovascular risk.")

# st.sidebar.header("Model Settings")
# model_name_ui = st.sidebar.selectbox("Choose model", ["Logistic Regression", "Random Forest"])
# model_param = "lr" if model_name_ui == "Logistic Regression" else "rf"
# st.sidebar.write(f"Using: *{model_name_ui}*")

st.header("Patient Details")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=50)
        gender = st.selectbox("Gender", ["Male", "Female"])
        height = st.number_input("Height (cm)", min_value=120, max_value=220, value=170)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)

    with col2:
        ap_hi = st.number_input("Systolic BP (ap_hi)", min_value=70, max_value=250, value=120)
        ap_lo = st.number_input("Diastolic BP (ap_lo)", min_value=40, max_value=150, value=80)
        cholesterol = st.selectbox("Cholesterol level", [1, 2, 3])
        gluc = st.selectbox("Glucose level", [1, 2, 3])

    st.markdown("### Lifestyle")
    col3, col4, col5 = st.columns(3)

    with col3:
        smoke = st.selectbox("Smokes?", ["No", "Yes"])
    with col4:
        alco = st.selectbox("Drinks alcohol?", ["No", "Yes"])
    with col5:
        active = st.selectbox("Physically active?", ["No", "Yes"])

    submitted = st.form_submit_button("Predict risk")

if submitted:
    gender_val = 1 if gender == "Female" else 2
    smoke_val = 1 if smoke == "Yes" else 0
    alco_val = 1 if alco == "Yes" else 0
    active_val = 1 if active == "Yes" else 0

    payload = {
        "age": int(age),
        "gender": int(gender_val),
        "height": float(height),
        "weight": float(weight),
        "ap_hi": float(ap_hi),
        "ap_lo": float(ap_lo),
        "cholesterol": int(cholesterol),
        "gluc": int(gluc),
        "smoke": int(smoke_val),
        "alco": int(alco_val),
        "active": int(active_val)
    }

    try:
        response = requests.post(
            API_URL,
            json=payload,
            # params={"model_name": model_param}
        )
        response.raise_for_status()
        result = response.json()

        pred = result["prediction"]
        proba = result["probability"]

        st.subheader("Result")

        if pred == 1:
            st.error("High risk of cardiovascular disease!")
        else:
            st.success("Low risk of cardiovascular disease")

        # st.write(f"Model used: *{result.get('model_used', model_name_ui)}*")
        st.write(f"Estimated risk probability: *{proba:.2f}*")

        bmi = weight / ((height / 100) ** 2)
        input_df = pd.DataFrame([{
            "age": age,
            "gender": gender_val,
            "height": height,
            "weight": weight,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "cholesterol": cholesterol,
            "gluc": gluc,
            "smoke": smoke_val,
            "alco": alco_val,
            "active": active_val,
            "bmi": bmi
        }])

        with st.expander("View input details"):
            st.write(input_df)

    except Exception as e:
        st.error(f"Error calling API: {e}")
