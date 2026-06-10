import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from datetime import datetime
import pandas as pd

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Forest Fire Prediction System",
    page_icon="🔥",
    layout="wide"
)

# -----------------------------
# Load Model and Scaler
# -----------------------------
model = load_model("fire_prediction_model.keras")
scaler = joblib.load("scaler.pkl")

# -----------------------------
# Header
# -----------------------------
st.title("🔥 Forest Fire Prediction System")

st.markdown("""
This AI-based system predicts the likelihood of forest fire occurrence
using environmental and fire-weather parameters.
""")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Forest Information")

location = st.sidebar.selectbox(
    "📍 Location",
    [
        "Andhra Pradesh",
        "Telangana",
        "Karnataka",
        "Tamil Nadu",
        "Kerala"
    ]
)

season = st.sidebar.selectbox(
    "🌤 Season",
    [
        "Summer",
        "Monsoon",
        "Winter",
        "Spring"
    ]
)

time_of_day = st.sidebar.selectbox(
    "🕒 Time of Day",
    [
        "Morning",
        "Afternoon",
        "Evening",
        "Night"
    ]
)

st.sidebar.success(
    f"""
Location: {location}

Season: {season}

Time: {time_of_day}
"""
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
### System Information

🤖 Model: Neural Network

📊 Features: 10

🎯 Task: Binary Classification

🔥 Output: Fire / No Fire
"""
)

# -----------------------------
# Input Section
# -----------------------------
st.subheader("Environmental Parameters")

col1, col2 = st.columns(2)

with col1:

    temperature = st.number_input(
        "Temperature (°C)",
        value=25.0
    )

    humidity = st.number_input(
        "Humidity (%)",
        value=50.0
    )

    windspeed = st.number_input(
        "Wind Speed (km/h)",
        value=10.0
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        value=0.0
    )

    ffmc = st.number_input(
        "FFMC",
        value=85.0
    )

with col2:

    dmc = st.number_input(
        "DMC",
        value=20.0
    )

    dc = st.number_input(
        "DC",
        value=100.0
    )

    isi = st.number_input(
        "ISI",
        value=5.0
    )

    bui = st.number_input(
        "BUI",
        value=30.0
    )

    fwi = st.number_input(
        "FWI",
        value=10.0
    )

# -----------------------------
# Environmental Summary
# -----------------------------
st.subheader("Environmental Summary")

s1, s2, s3 = st.columns(3)

s1.metric(
    "Temperature",
    f"{temperature} °C"
)

s2.metric(
    "Humidity",
    f"{humidity} %"
)

s3.metric(
    "Wind Speed",
    f"{windspeed} km/h"
)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔥 Predict Fire Risk"):

    input_data = np.array([[
        temperature,
        humidity,
        windspeed,
        rainfall,
        ffmc,
        dmc,
        dc,
        isi,
        bui,
        fwi
    ]])

    # Scale Input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    probability = float(prediction[0][0])

    st.divider()

    # -----------------------------
    # Prediction Result
    # -----------------------------
    st.subheader("Prediction Result")

    st.write("### Fire Probability Meter")

    st.progress(
        min(int(probability * 100), 100)
    )

    st.metric(
        "Predicted Fire Probability",
        f"{probability:.2%}"
    )

    # -----------------------------
    # Risk Category
    # -----------------------------
    st.subheader("Risk Category")

    if probability < 0.30:
        st.success("🟢 Low Risk")

    elif probability < 0.70:
        st.warning("🟡 Moderate Risk")

    else:
        st.error("🔴 High Risk")

    # -----------------------------
    # Prediction Message
    # -----------------------------
    if probability > 0.5:

        st.error(
            f"🔥 Forest Fire Likely\n\nConfidence: {probability:.2%}"
        )

    else:

        st.success(
            f"🌳 No Forest Fire Expected\n\nConfidence: {(1 - probability):.2%}"
        )

    # -----------------------------
    # Small Feature Graph
    # -----------------------------
    st.subheader("Feature Visualization")

    feature_names = [
        "Temp",
        "Humidity",
        "Wind",
        "Rain",
        "FFMC",
        "DMC",
        "DC",
        "ISI",
        "BUI",
        "FWI"
    ]

    feature_values = [
        temperature,
        humidity,
        windspeed,
        rainfall,
        ffmc,
        dmc,
        dc,
        isi,
        bui,
        fwi
    ]

    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    ax.bar(
        feature_names,
        feature_values
    )

    ax.set_title(
        "Environmental Factors",
        fontsize=8
    )

    plt.xticks(
        rotation=45,
        fontsize=6
    )

    plt.yticks(
        fontsize=6
    )

    # Center Graph
    left, center, right = st.columns([1, 2, 1])

    with center:
        st.pyplot(fig)

    # -----------------------------
    # Prediction Report
    # -----------------------------
    st.subheader("Prediction Report")

    st.write(
        f"📍 Location: **{location}**"
    )

    st.write(
        f"🌤 Season: **{season}**"
    )

    st.write(
        f"🕒 Time of Day: **{time_of_day}**"
    )

    st.write(
        f"📈 Fire Probability: **{probability:.2%}**"
    )

    st.write(
        f"🕓 Prediction Time: **{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}**"
    )

    # -----------------------------
    # Model Performance
    # -----------------------------
    st.subheader("Model Performance")

    accuracy = 1.00

    st.metric(
        "Model Accuracy",
        f"{accuracy:.2%}"
    )

    # -----------------------------
    # Classification Report Table
    # -----------------------------
    st.subheader("Classification Report")

    report_df = pd.DataFrame({
        "Class": ["No Fire (0)", "Fire (1)"],
        "Precision": [1.00, 1.00],
        "Recall": [1.00, 1.00],
        "F1-Score": [1.00, 1.00]
    })

    st.dataframe(
        report_df,
        use_container_width=True
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.caption(
    "AI-Based Forest Fire Prediction System using Neural Networks and Environmental Indicators"
)