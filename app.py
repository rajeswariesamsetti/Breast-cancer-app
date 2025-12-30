# app.py

import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction",
    layout="centered"
)

# -------------------------------
# Custom CSS Styling
# -------------------------------
st.markdown("""
<style>
/* Main background */
.main {
    background-color: #f7f9fc;
}

/* Title */
h1 {
    color: #b30000;
    text-align: center;
}

/* Subheaders */
h2, h3 {
    color: #333333;
}

/* Buttons */
.stButton > button {
    background-color: #b30000;
    color: white;
    border-radius: 8px;
    padding: 10px 25px;
    font-size: 16px;
    border: none;
}

.stButton > button:hover {
    background-color: #800000;
    color: white;
}

/* Input boxes */
div[data-baseweb="input"] > div {
    border-radius: 8px;
}

/* Result box */
.result-box {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    text-align: center;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title & description
# -------------------------------
st.title("🩺 Breast Cancer Prediction")
st.write("Enter tumor feature values to predict whether the tumor is **Benign** or **Malignant**.")

# -------------------------------
# Load model & scaler
# -------------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("BC_model.h5")
    with open("BC_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# -------------------------------
# Input fields
# -------------------------------
st.subheader("🔢 Input Features")

input_data = {
    'radius_mean': st.number_input('radius_mean', value=14.0),
    'texture_mean': st.number_input('texture_mean', value=20.0),
    'perimeter_mean': st.number_input('perimeter_mean', value=90.0),
    'area_mean': st.number_input('area_mean', value=600.0),
    'smoothness_mean': st.number_input('smoothness_mean', value=0.1),
    'compactness_mean': st.number_input('compactness_mean', value=0.15),
    'concavity_mean': st.number_input('concavity_mean', value=0.2),
    'concave points_mean': st.number_input('concave points_mean', value=0.1),
    'symmetry_mean': st.number_input('symmetry_mean', value=0.2),
    'fractal_dimension_mean': st.number_input('fractal_dimension_mean', value=0.06),

    'radius_se': st.number_input('radius_se', value=0.2),
    'texture_se': st.number_input('texture_se', value=1.0),
    'perimeter_se': st.number_input('perimeter_se', value=1.5),
    'area_se': st.number_input('area_se', value=20.0),
    'smoothness_se': st.number_input('smoothness_se', value=0.005),
    'compactness_se': st.number_input('compactness_se', value=0.02),
    'concavity_se': st.number_input('concavity_se', value=0.03),
    'concave points_se': st.number_input('concave points_se', value=0.01),
    'symmetry_se': st.number_input('symmetry_se', value=0.03),
    'fractal_dimension_se': st.number_input('fractal_dimension_se', value=0.004),

    'radius_worst': st.number_input('radius_worst', value=16.0),
    'texture_worst': st.number_input('texture_worst', value=25.0),
    'perimeter_worst': st.number_input('perimeter_worst', value=105.0),
    'area_worst': st.number_input('area_worst', value=800.0),
    'smoothness_worst': st.number_input('smoothness_worst', value=0.12),
    'compactness_worst': st.number_input('compactness_worst', value=0.2),
    'concavity_worst': st.number_input('concavity_worst', value=0.3),
    'concave points_worst': st.number_input('concave points_worst', value=0.15),
    'symmetry_worst': st.number_input('symmetry_worst', value=0.25),
    'fractal_dimension_worst': st.number_input('fractal_dimension_worst', value=0.08),
}

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict"):
    input_df = pd.DataFrame([input_data])

    if list(input_df.columns) != list(scaler.feature_names_in_):
        st.error("Feature mismatch between input and trained scaler.")
        st.stop()

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0][0]

    predicted_class = "Malignant 🚨" if prediction > 0.5 else "Benign ✅"

    st.markdown(f"""
    <div class="result-box">
        <h2>Prediction Result</h2>
        <h3>{predicted_class}</h3>
        <p><b>Probability:</b> {prediction:.4f}</p>
    </div>
    """, unsafe_allow_html=True)
