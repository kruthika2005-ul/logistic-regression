import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Page settings
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="📊",
    layout="centered"
)

# Title
st.title("📊 Breast Cancer Prediction App")

st.write("Logistic Regression Model")

# Load dataset
df = pd.read_excel("logistic_regression_dataset.xlsx")

# Features and target
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# Show accuracy
st.subheader("Model Performance")
st.write(f"### Accuracy Score: {accuracy:.2f}")

# Input section
st.subheader("Enter Input Features")

# User inputs
radius_mean = st.number_input(
    "Radius Mean",
    min_value=0.0,
    value=14.0,
    step=0.1
)

texture_mean = st.number_input(
    "Texture Mean",
    min_value=0.0,
    value=20.0,
    step=0.1
)

perimeter_mean = st.number_input(
    "Perimeter Mean",
    min_value=0.0,
    value=90.0,
    step=0.1
)

area_mean = st.number_input(
    "Area Mean",
    min_value=0.0,
    value=600.0,
    step=1.0
)

smoothness_mean = st.number_input(
    "Smoothness Mean",
    min_value=0.0,
    value=0.1,
    step=0.01
)

# Predict button
if st.button("Predict"):

    # Create dataframe
    input_data = pd.DataFrame({
        'radius_mean': [radius_mean],
        'texture_mean': [texture_mean],
        'perimeter_mean': [perimeter_mean],
        'area_mean': [area_mean],
        'smoothness_mean': [smoothness_mean]
    })

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    # Probability
    probability = model.predict_proba(input_scaled)

    # Result
    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Benign Tumor")
    else:
        st.error("❌ Malignant Tumor")

    # Probability
    st.subheader("Prediction Probability")

    st.write({
        "Malignant": round(probability[0][0] * 100, 2),
        "Benign": round(probability[0][1] * 100, 2)
    })

# Footer
st.markdown("---")
st.caption("Developed using Streamlit and Logistic Regression")