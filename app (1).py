
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Healthcare System",
    page_icon="🏥",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL AND LABEL ENCODER
# --------------------------------------------------

MODEL_PATH = "healthcare_model.pkl"
ENCODER_PATH = "label_encoder.pkl"

model = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🏥 Smart Healthcare System")
st.subheader("Machine Learning and Big Data Based Healthcare Monitoring")

st.write(
    "This system uses patient vital signs and a Random Forest "
    "machine learning model to predict the disease category."
)

st.warning(
    "⚠️ This application is an educational/demo system based on a "
    "synthetic dataset. Predictions are not medical diagnoses."
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "🩺 Disease Prediction",
        "📋 Patient History",
        "📊 Doctor Dashboard"
    ]
)

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if page == "🏠 Home":

    st.header("Welcome to the Smart Healthcare System")

    st.write("""
    The Smart Healthcare System demonstrates how Machine Learning
    can be used to analyze patient health parameters and predict
    disease categories.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Machine Learning Model", "Random Forest")

    with col2:
        st.metric("Input Parameters", "5")

    with col3:
        st.metric("Dataset Type", "Synthetic")

    st.markdown("### System Inputs")

    st.write("""
    • Heart Rate (bpm)
    • SpO2 Level (%)
    • Systolic Blood Pressure (mmHg)
    • Diastolic Blood Pressure (mmHg)
    • Body Temperature (°C)
    """)

    st.markdown("### System Output")

    st.write("""
    The system provides:

    • Predicted disease category
    • Prediction confidence
    • Class probabilities
    • Basic health alerts
    • Risk level
    • General recommendations
    • Patient history
    • Dashboard statistics
    """)


# --------------------------------------------------
# DISEASE PREDICTION PAGE
# --------------------------------------------------

elif page == "🩺 Disease Prediction":

    st.header("Patient Disease Prediction")

    st.write(
        "Enter the patient's vital signs below."
    )

    col1, col2 = st.columns(2)

    with col1:

        patient_name = st.text_input(
            "Patient Name",
            value="Patient"
        )

        heart_rate = st.number_input(
            "Heart Rate (bpm)",
            min_value=30.0,
            max_value=200.0,
            value=80.0
        )

        spo2 = st.number_input(
            "SpO2 Level (%)",
            min_value=50.0,
            max_value=100.0,
            value=98.0
        )

        systolic_bp = st.number_input(
            "Systolic Blood Pressure (mmHg)",
            min_value=60.0,
            max_value=250.0,
            value=120.0
        )

    with col2:

        diastolic_bp = st.number_input(
            "Diastolic Blood Pressure (mmHg)",
            min_value=30.0,
            max_value=150.0,
            value=80.0
        )

        temperature = st.number_input(
            "Body Temperature (°C)",
            min_value=30.0,
            max_value=45.0,
            value=37.0
        )

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    if st.button("🔍 Predict Disease"):

        input_data = pd.DataFrame(
            [[
                heart_rate,
                spo2,
                systolic_bp,
                diastolic_bp,
                temperature
            ]],
            columns=[
                "Heart Rate (bpm)",
                "SpO2 Level (%)",
                "Systolic Blood Pressure (mmHg)",
                "Diastolic Blood Pressure (mmHg)",
                "Body Temperature (°C)"
            ]
        )

        prediction = model.predict(input_data)

        disease = le.inverse_transform(prediction)[0]

        probabilities = model.predict_proba(input_data)[0]

        confidence = np.max(probabilities) * 100

        # --------------------------------------------------
        # DISPLAY RESULT
        # --------------------------------------------------

        st.success(f"Predicted Disease Category: **{disease}**")

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        # --------------------------------------------------
        # PROBABILITY TABLE
        # --------------------------------------------------

        probability_df = pd.DataFrame({
            "Disease": le.classes_,
            "Probability (%)": probabilities * 100
        })

        probability_df = probability_df.sort_values(
            by="Probability (%)",
            ascending=False
        ).reset_index(drop=True)

        st.subheader("Prediction Probabilities")

        st.dataframe(
            probability_df,
            use_container_width=True
        )

        st.bar_chart(
            probability_df.set_index("Disease")
        )

        # --------------------------------------------------
        # BASIC DEMO ALERTS
        # --------------------------------------------------

        st.subheader("Health Monitoring Alerts")

        alerts = []

        if spo2 < 90:
            alerts.append("Low SpO2 level detected.")

        if heart_rate < 50 or heart_rate > 120:
            alerts.append("Heart rate is outside the configured monitoring range.")

        if systolic_bp > 140:
            alerts.append("Elevated systolic blood pressure detected.")

        if diastolic_bp > 90:
            alerts.append("Elevated diastolic blood pressure detected.")

        if temperature > 38:
            alerts.append("Elevated body temperature detected.")

        if len(alerts) == 0:
            st.info("No configured monitoring alerts detected.")
        else:
            for alert in alerts:
                st.warning("⚠️ " + alert)

        # --------------------------------------------------
        # RISK LEVEL
        # --------------------------------------------------

        alert_count = len(alerts)

        if alert_count == 0:
            risk = "Low"
        elif alert_count <= 2:
            risk = "Moderate"
        else:
            risk = "High"

        st.subheader("Risk Level")

        if risk == "Low":
            st.success("🟢 Low Risk")

        elif risk == "Moderate":
            st.warning("🟡 Moderate Risk")

        else:
            st.error("🔴 High Risk")

        # --------------------------------------------------
        # GENERAL RECOMMENDATION
        # --------------------------------------------------

        st.subheader("General Recommendation")

        if risk == "Low":
            st.write(
                "Continue regular health monitoring and maintain "
                "healthy lifestyle practices."
            )

        elif risk == "Moderate":
            st.write(
                "Monitor the patient's vital signs regularly and "
                "consider professional medical evaluation if symptoms persist."
            )

        else:
            st.write(
                "Seek appropriate professional medical attention, "
                "especially if the patient has concerning symptoms."
            )

        # --------------------------------------------------
        # SAVE HISTORY
        # --------------------------------------------------

        history_file = "patient_history.csv"

        record = pd.DataFrame([{
            "Patient Name": patient_name,
            "Heart Rate (bpm)": heart_rate,
            "SpO2 Level (%)": spo2,
            "Systolic Blood Pressure (mmHg)": systolic_bp,
            "Diastolic Blood Pressure (mmHg)": diastolic_bp,
            "Body Temperature (°C)": temperature,
            "Predicted Disease": disease,
            "Confidence (%)": confidence,
            "Risk Level": risk
        }])

        if os.path.exists(history_file):

            history = pd.read_csv(history_file)

            history = pd.concat(
                [history, record],
                ignore_index=True
            )

        else:

            history = record

        history.to_csv(
            history_file,
            index=False
        )

        st.success("Patient prediction saved to history.")


# --------------------------------------------------
# PATIENT HISTORY PAGE
# --------------------------------------------------

elif page == "📋 Patient History":

    st.header("📋 Patient History")

    history_file = "patient_history.csv"

    if os.path.exists(history_file):

        history = pd.read_csv(history_file)

        st.dataframe(
            history,
            use_container_width=True
        )

        st.subheader("Search Patient")

        search_name = st.text_input(
            "Enter patient name"
        )

        if search_name:

            result = history[
                history["Patient Name"]
                .astype(str)
                .str.contains(
                    search_name,
                    case=False,
                    na=False
                )
            ]

            st.dataframe(
                result,
                use_container_width=True
            )

    else:

        st.info(
            "No patient history available yet. "
            "Make a prediction first."
        )


# --------------------------------------------------
# DOCTOR DASHBOARD
# --------------------------------------------------

elif page == "📊 Doctor Dashboard":

    st.header("📊 Doctor Dashboard")

    history_file = "patient_history.csv"

    if os.path.exists(history_file):

        history = pd.read_csv(history_file)

        # --------------------------------------------------
        # DASHBOARD METRICS
        # --------------------------------------------------

        total_patients = len(history)

        disease_count = history["Predicted Disease"].nunique()

        high_risk = len(
            history[
                history["Risk Level"] == "High"
            ]
        )

        average_confidence = history[
            "Confidence (%)"
        ].mean()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Predictions",
                total_patients
            )

        with col2:
            st.metric(
                "Disease Categories",
                disease_count
            )

        with col3:
            st.metric(
                "High Risk Cases",
                high_risk
            )

        with col4:
            st.metric(
                "Average Confidence",
                f"{average_confidence:.2f}%"
            )

        # --------------------------------------------------
        # DISEASE DISTRIBUTION
        # --------------------------------------------------

        st.subheader("Disease Distribution")

        disease_distribution = (
            history["Predicted Disease"]
            .value_counts()
        )

        st.bar_chart(
            disease_distribution
        )

        # --------------------------------------------------
        # RISK DISTRIBUTION
        # --------------------------------------------------

        st.subheader("Risk Level Distribution")

        risk_distribution = (
            history["Risk Level"]
            .value_counts()
        )

        st.bar_chart(
            risk_distribution
        )

        # --------------------------------------------------
        # PATIENT DATA
        # --------------------------------------------------

        st.subheader("Patient Records")

        st.dataframe(
            history,
            use_container_width=True
        )

    else:

        st.info(
            "No patient data available. "
            "Generate predictions first."
        )
