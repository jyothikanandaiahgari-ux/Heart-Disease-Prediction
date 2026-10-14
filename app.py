import streamlit as st
import joblib
import pandas as pd
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER

# ============================================================
# PREDICTION HISTORY
# ============================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []
# ============================================================
# LOAD SAVED MODEL FILES
# ============================================================

model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("heart_disease_scaler.pkl")
encoders = joblib.load("heart_disease_encoders.pkl")

# ============================================================
# USER-FRIENDLY LABELS
# ============================================================

sex_labels = {
    "F": "Female",
    "M": "Male"
}

chest_pain_labels = {
    "ASY": "Asymptomatic",
    "ATA": "Atypical Angina",
    "NAP": "Non-Anginal Pain",
    "TA": "Typical Angina"
}

resting_ecg_labels = {
    "LVH": "Left Ventricular Hypertrophy",
    "Normal": "Normal",
    "ST": "ST-T Wave Abnormality"
}

exercise_angina_labels = {
    "N": "No",
    "Y": "Yes"
}

st_slope_labels = {
    "Down": "Downsloping",
    "Flat": "Flat",
    "Up": "Upsloping"
}

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("❤️ Heart Disease")
st.sidebar.markdown("### Prediction System")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔍 Prediction",
        "📘 About"
    ]
)
# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="main-title">'
        '❤️ Heart Disease Prediction System'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Machine Learning Based Heart Disease Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Introduction
    st.markdown("## 👋 Welcome")

    st.write(
        "Welcome to the Heart Disease Prediction System. "
        "This application uses machine learning to classify "
        "whether a person is predicted to have heart disease "
        "based on selected health and heart-related parameters."
    )

    # Project highlights
    st.markdown("## 🚀 Project Highlights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🤖 Machine Learning")
        st.write(
            "Multiple classification algorithms were trained "
            "and compared to identify the selected model."
        )

    with col2:
        st.markdown("### 🏆 Selected Model")
        st.write(
            "**SVM** was selected based on the highest test "
            "accuracy obtained in this project."
        )

    with col3:
        st.markdown("### 📊 Accuracy")
        st.write(
            "The selected SVM model achieved "
            "**86.41% test accuracy**."
        )

    # How it works
    st.markdown("## 🔄 How It Works")

    st.write(
        "1. Enter the patient's health information."
    )

    st.write(
        "2. The application converts categorical inputs "
        "into the format expected by the trained model."
    )

    st.write(
        "3. The same scaling process used during model "
        "training is applied to the input."
    )

    st.write(
        "4. The trained SVM model generates a prediction."
    )

    st.write(
        "5. The application displays the prediction and "
        "model-estimated probability."
    )

    st.markdown("---")

    st.info(
        "💡 Use the **Prediction** option in the sidebar "
        "to enter patient information and generate a prediction."
    )
st.sidebar.markdown("---")

st.sidebar.write("### 🤖 Model Information")

st.sidebar.write("**Model:** SVM")
st.sidebar.write("**Test Accuracy:** 86.41%")

st.sidebar.markdown("---")

st.sidebar.caption(
    "Academic Machine Learning Project"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

/* Main title */
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #9ca3af;
    margin-bottom: 30px;
}

/* Section headings */
.section-title {
    font-size: 24px;
    font-weight: 600;
    margin-top: 25px;
    margin-bottom: 15px;
}

/* Info cards */
.info-card {
    padding: 18px;
    border-radius: 12px;
    background-color: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 15px;
}

/* Prediction card */
.prediction-card {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin-top: 25px;
    margin-bottom: 20px;
}

/* Small text */
.small-text {
    color: #9ca3af;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">❤️ Heart Disease Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Based Heart Disease Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "Enter the patient's health information below to generate a "
    "machine-learning based prediction."
)
# ============================================================
# PREDICTION PAGE
# ============================================================

if page == "🔍 Prediction":

    # ============================================================
    # PATIENT INFORMATION
    # ============================================================

    st.markdown(
        '<div class="section-title">👤 Patient Information</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=40
        )

        sex = st.selectbox(
            "Gender",
            encoders["sex"].classes_,
            format_func=lambda x: sex_labels[x]
        )

        chest_pain = st.selectbox(
            "Chest Pain Type",
            encoders["chest_pain"].classes_,
            format_func=lambda x: chest_pain_labels[x]
        )


    with col2:

        resting_bp = st.number_input(
            "Resting Blood Pressure",
            min_value=0,
            max_value=300,
            value=120
        )

        cholesterol = st.number_input(
            "Cholesterol",
            min_value=0,
            max_value=700,
            value=200
        )

        fasting_bs = st.selectbox(
            "Fasting Blood Sugar",
            [0, 1]
        )


    # ============================================================
    # HEART & ECG INFORMATION
    # ============================================================

    st.markdown(
        '<div class="section-title">🫀 Heart & ECG Information</div>',
        unsafe_allow_html=True
    )

    col3, col4 = st.columns(2)

    with col3:

        resting_ecg = st.selectbox(
            "Resting ECG",
            encoders["resting_ecg"].classes_,
            format_func=lambda x: resting_ecg_labels[x]
        )

        max_hr = st.number_input(
            "Maximum Heart Rate",
            min_value=0,
            max_value=250,
            value=150
        )


    with col4:

        exercise_angina = st.selectbox(
            "Exercise Angina",
            encoders["exercise_angina"].classes_,
            format_func=lambda x: exercise_angina_labels[x]
        )

        oldpeak = st.number_input(
            "Oldpeak",
            min_value=-10.0,
            max_value=20.0,
            value=1.0,
            step=0.1
        )

        st_slope = st.selectbox(
            "ST Slope",
            encoders["st_slope"].classes_,
            format_func=lambda x: st_slope_labels[x]
        )


    # ============================================================
    # PREDICTION BUTTON
    # ============================================================

    st.markdown("<br>", unsafe_allow_html=True)

    # ========================================================
    # ACTION BUTTONS
    # ========================================================

    col_button1, col_button2 = st.columns(2)

    with col_button1:

        predict_button = st.button(
            "🔍 Predict Heart Disease",
            use_container_width=True,
            type="primary"
        )

    with col_button2:

        reset_button = st.button(
            "🔄 Reset Form",
            use_container_width=True
        )

        # ========================================================
        # RESET FORM
        # ========================================================

        if reset_button:

            st.rerun()


    # ============================================================
    # PREDICTION
    # ============================================================

    if predict_button:

        # --------------------------------------------------------
        # Encode categorical values
        # --------------------------------------------------------

        sex_encoded = encoders["sex"].transform([sex])[0]

        chest_pain_encoded = encoders["chest_pain"].transform(
            [chest_pain]
        )[0]

        resting_ecg_encoded = encoders["resting_ecg"].transform(
            [resting_ecg]
        )[0]

        exercise_angina_encoded = encoders["exercise_angina"].transform(
            [exercise_angina]
        )[0]

        st_slope_encoded = encoders["st_slope"].transform(
            [st_slope]
        )[0]


        # --------------------------------------------------------
        # Create input
        # --------------------------------------------------------

        user_input = [[
            age,
            sex_encoded,
            chest_pain_encoded,
            resting_bp,
            cholesterol,
            fasting_bs,
            resting_ecg_encoded,
            max_hr,
            exercise_angina_encoded,
            oldpeak,
            st_slope_encoded
        ]]


        # --------------------------------------------------------
        # Convert to DataFrame
        # --------------------------------------------------------

        input_df = pd.DataFrame(
            user_input,
            columns=[
                "Age",
                "Sex",
                "ChestPainType",
                "RestingBP",
                "Cholesterol",
                "FastingBS",
                "RestingECG",
                "MaxHR",
                "ExerciseAngina",
                "Oldpeak",
                "ST_Slope"
            ]
        )


        # --------------------------------------------------------
        # Scale input
        # --------------------------------------------------------

        input_scaled = scaler.transform(input_df)


        # --------------------------------------------------------
        # Prediction
        # --------------------------------------------------------

        prediction = model.predict(input_scaled)


        # --------------------------------------------------------
        # Probability
        # --------------------------------------------------------

        probabilities = model.predict_proba(input_scaled)

        predicted_probability = probabilities[0][prediction[0]]

        # ============================================================
        # SAVE PREDICTION TO HISTORY
        # ============================================================

        prediction_text = (
            "Person Having Heart Disease"
            if prediction[0] == 1
            else "Person Not Having Heart Disease"
        )

        history_record = {
            "Time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Age": age,
            "Sex": sex_labels[sex],
            "Chest Pain Type": chest_pain_labels[chest_pain],
            "Resting BP": resting_bp,
            "Cholesterol": cholesterol,
            "Prediction": prediction_text,
            "Probability (%)": round(predicted_probability * 100, 2)
        }

        st.session_state.prediction_history.append(history_record)

        # ============================================================
        # CLASS PROBABILITIES
        # ============================================================

        prob_no_disease = probabilities[0][0] * 100
        prob_disease = probabilities[0][1] * 100

        st.markdown("---")

        st.markdown(
            '<div class="section-title">📊 Prediction Probability</div>',
            unsafe_allow_html=True
        )

        probability_data = pd.DataFrame({
            "Prediction": [
                "Person Not Having Heart Disease",
                "Person Having Heart Disease"
            ],
            "Probability (%)": [
                prob_no_disease,
                prob_disease
            ]
        })

        st.dataframe(
            probability_data,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            probability_data.set_index("Prediction")
        )

        # ========================================================
        # PATIENT INPUT SUMMARY
        # ========================================================

        st.markdown(
            '<div class="section-title">📋 Patient Input Summary</div>',
            unsafe_allow_html=True
        )

        input_summary = pd.DataFrame({
            "Parameter": [
                "Age",
                "Gender",
                "Chest Pain Type",
                "Resting Blood Pressure",
                "Cholesterol",
                "Fasting Blood Sugar",
                "Resting ECG",
                "Maximum Heart Rate",
                "Exercise Angina",
                "Oldpeak",
                "ST Slope"
            ],

            "Value": [
                age,
                sex_labels[sex],
                chest_pain_labels[chest_pain],
                resting_bp,
                cholesterol,
                "High (1)" if fasting_bs == 1 else "Normal (0)",
                resting_ecg_labels[resting_ecg],
                max_hr,
                exercise_angina_labels[exercise_angina],
                oldpeak,
                st_slope_labels[st_slope]
            ]
        })

        st.dataframe(
            input_summary,
            use_container_width=True,
            hide_index=True
        )
        # ========================================================
        # RESULT DASHBOARD
        # ========================================================

        st.markdown(
            '<div class="section-title">📊 Prediction Result</div>',
            unsafe_allow_html=True
        )

        probability_percent = predicted_probability * 100


        # --------------------------------------------------------
        # Prediction Status
        # --------------------------------------------------------

        if prediction[0] == 0:

            st.success(
                "✅ Prediction: Person Not Having Heart Disease"
            )

            result_message = (
                "The model classified this input as not having "
                "heart disease."
            )

        else:

            st.error(
                "⚠️ Prediction: Person Having Heart Disease"
            )

            result_message = (
                "The model classified this input as having "
                "heart disease."
            )


        # --------------------------------------------------------
        # Result Metrics
        # --------------------------------------------------------

        col5, col6 = st.columns(2)

        with col5:

            st.metric(
                "Model Estimated Probability",
                f"{probability_percent:.2f}%"
            )

        with col6:

            st.metric(
                "Model Used",
                "SVM"
            )


        # --------------------------------------------------------
        # Probability Indicator
        # --------------------------------------------------------

        st.write("### 📈 Model Estimated Probability")

        st.progress(
            predicted_probability
        )

        st.caption(
            f"Estimated probability for the predicted class: "
            f"{probability_percent:.2f}%"
        )


        # --------------------------------------------------------
        # Result Explanation
        # --------------------------------------------------------

        st.info(result_message)


        # --------------------------------------------------------
        # Important Note
        # --------------------------------------------------------

        st.warning(
            "This probability is produced by the machine-learning "
            "model for this input. It should not be interpreted as "
            "a patient's actual medical risk or as a medical diagnosis."
        )

        # ========================================================
        # MODEL INFORMATION
        # ========================================================

        with st.expander("ℹ️ About the Machine Learning Model"):

            st.write(
                "The prediction is generated using a Support Vector "
                "Machine (SVM) classification model."
            )

            st.write(
                "The model was trained using 11 health-related "
                "features and the input data is standardized using "
                "the same scaler used during training."
            )

            st.write(
                "The selected SVM model achieved approximately "
                "86.41% accuracy on the project's test dataset."
            )

            # ============================================================
            # DOWNLOAD PREDICTION REPORT - PDF
            # ============================================================
            
            st.markdown("---")
            
            st.markdown(
                '<div class="section-title">📄 Prediction Report</div>',
                unsafe_allow_html=True
            )
            
            # Convert prediction to readable text
            if prediction[0] == 0:
                prediction_text = "Person Not Having Heart Disease"
            else:
                prediction_text = "Person Having Heart Disease"
            
            
            # ============================================================
            # CREATE PDF
            # ============================================================
            
            pdf_buffer = BytesIO()
            
            doc = SimpleDocTemplate(
                pdf_buffer,
                pagesize=A4,
                rightMargin=40,
                leftMargin=40,
                topMargin=40,
                bottomMargin=40
            )
            
            styles = getSampleStyleSheet()
            
            title_style = styles["Title"]
            title_style.alignment = TA_CENTER
            
            heading_style = styles["Heading2"]
            normal_style = styles["BodyText"]
            
            
            # IMPORTANT: Create the elements list
            elements = []
            
            
            # ============================================================
            # PDF TITLE
            # ============================================================
            
            elements.append(
                Paragraph(
                    "Heart Disease Prediction Report",
                    title_style
                )
            )
            
            elements.append(Spacer(1, 20))
            
            
            # ============================================================
            # PATIENT INFORMATION
            # ============================================================
            
            elements.append(
                Paragraph(
                    "Patient Information",
                    heading_style
                )
            )
            
            patient_data = [
                ["Parameter", "Value"],
            
                ["Age", str(age)],
            
                ["Sex", sex_labels[sex]],
            
                ["Chest Pain Type",
                chest_pain_labels[chest_pain]],
            
                ["Resting Blood Pressure",
                str(resting_bp)],
            
                ["Cholesterol",
                str(cholesterol)],
            
                ["Fasting Blood Sugar",
                "High (1)" if fasting_bs == 1 else "Normal (0)"],
            
                ["Resting ECG",
                resting_ecg_labels[resting_ecg]],
            
                ["Maximum Heart Rate",
                str(max_hr)],
            
                ["Exercise Angina",
                exercise_angina_labels[exercise_angina]],
            
                ["Oldpeak",
                str(oldpeak)],
            
                ["ST Slope",
                st_slope_labels[st_slope]]
            ]
            
            patient_table = Table(
                patient_data,
                colWidths=[220, 250]
            )
            
            patient_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("PADDING", (0, 0), (-1, -1), 7),
                ])
            )
            
            elements.append(patient_table)
            
            elements.append(Spacer(1, 20))
            
            
            # ============================================================
            # PREDICTION RESULT
            # ============================================================
            
            elements.append(
                Paragraph(
                    "Prediction Result",
                    heading_style
                )
            )
            
            result_data = [
                ["Parameter", "Result"],
            
                ["Prediction",
                prediction_text],
            
                ["Probability - No Heart Disease",
                f"{prob_no_disease:.2f}%"],
            
                ["Probability - Heart Disease",
                f"{prob_disease:.2f}%"],
            
                ["Model Used",
                "Support Vector Machine (SVM)"],
            
                ["Test Accuracy",
                "86.41%"]
            ]
            
            result_table = Table(
                result_data,
                colWidths=[220, 250]
            )
            
            result_table.setStyle(
                TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("PADDING", (0, 0), (-1, -1), 7),
                ])
            )
            
            elements.append(result_table)
            
            elements.append(Spacer(1, 25))
            
            
            # ============================================================
            # DISCLAIMER
            # ============================================================
            
            elements.append(
                Paragraph(
                    "<b>Important Notice:</b> This report is generated by "
                    "a machine-learning model for academic and educational "
                    "purposes. The prediction is not a medical diagnosis "
                    "and should not be considered a substitute for advice "
                    "from a qualified healthcare professional.",
                    normal_style
                )
            )
            
            
            # ============================================================
            # BUILD PDF
            # ============================================================
            
            doc.build(elements)
            
            pdf_data = pdf_buffer.getvalue()
            
            
            # ============================================================
            # DOWNLOAD BUTTON
            # ============================================================
            
            st.download_button(
                label="📄 Download Prediction Report (PDF)",
                data=pdf_data,
                file_name="heart_disease_prediction_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

            # ============================================================
            # PREDICTION HISTORY
            # ============================================================

            st.markdown("---")

            st.markdown(
                '<div class="section-title">🕘 Prediction History</div>',
                unsafe_allow_html=True
            )

            if len(st.session_state.prediction_history) > 0:

                history_df = pd.DataFrame(
                    st.session_state.prediction_history
                )

                st.dataframe(
                    history_df,
                    use_container_width=True,
                    hide_index=True
                )

                if st.button(
                    "🗑️ Clear Prediction History",
                    use_container_width=True
                ):
                    st.session_state.prediction_history = []
                    st.rerun()

            else:

                st.info(
                    "No prediction history available yet. "
                    "Make a prediction to see it here."
                )

# ============================================================
# ABOUT PAGE
# ============================================================

if page == "📘 About":
    # ============================================================
    # ABOUT THE PROJECT
    # ============================================================

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📘 About the Project</div>',
        unsafe_allow_html=True
    )

    with st.expander("🔎 View Project Details"):

        st.write("### ❤️ Heart Disease Prediction System")

        st.write(
            "This project is a machine-learning based system that "
            "predicts whether a person is classified as having heart "
            "disease based on selected health and heart-related "
            "features."
        )

        st.write("### 📊 Dataset Information")

        st.write(
            "The model was trained using the heart disease dataset "
            "used in this project. The dataset contains 11 input "
            "features related to patient health and heart condition."
        )

        st.write("### 🧾 Input Features")

        feature_list = [
            "Age",
            "Sex",
            "Chest Pain Type",
            "Resting Blood Pressure",
            "Cholesterol",
            "Fasting Blood Sugar",
            "Resting ECG",
            "Maximum Heart Rate",
            "Exercise Angina",
            "Oldpeak",
            "ST Slope"
        ]

        for feature in feature_list:
            st.write(f"• {feature}")

        st.write("### 🤖 Machine Learning Models")

        models_used = [
            "Logistic Regression",
            "Support Vector Machine (SVM)",
            "K-Nearest Neighbors (KNN)",
            "Decision Tree",
            "Random Forest",
            "XGBoost"
        ]

        for model_name in models_used:
            st.write(f"• {model_name}")

        st.write("### 🏆 Selected Model")

        st.write(
            "The Support Vector Machine (SVM) model was selected "
            "based on the highest accuracy obtained during the "
            "model comparison."
        )

        st.write(
            "**Test Dataset Accuracy:** 86.41%"
        )

        st.write("### 🎓 Project Purpose")

        st.write(
            "This project is developed for academic and educational "
            "purposes to demonstrate the application of machine "
            "learning in healthcare-related prediction."
        )
        # ============================================================
        # MODEL PERFORMANCE COMPARISON
        #============================================================
        
        st.markdown(
            '<div class="section-title">📊 Model Performance Comparison</div>',
            unsafe_allow_html=True
        )
        
        model_performance = pd.DataFrame({
            "Model": [
                "Logistic Regression",
                "SVM",
                "KNN",
                "Decision Tree",
                "Random Forest",
                "XGBoost"
            ],
        "Accuracy": [
                83.15,
                86.41,
                83.70,
                76.09,
                84.24,
                84.24
            ]
        })
        
        st.dataframe(
            model_performance,
            use_container_width=True,
            hide_index=True
        )
        
        st.write("### 📈 Accuracy Comparison")
        
        st.bar_chart(
            model_performance.set_index("Model")
        )

        
# ============================================================
# DISCLAIMER
# ============================================================

st.markdown("---")

st.caption(
    "⚠️ Educational Project: This application is intended for "
    "academic and demonstration purposes only. The prediction "
    "is generated by a machine-learning model and should not "
    "be considered a medical diagnosis or a substitute for "
    "professional medical advice."
)