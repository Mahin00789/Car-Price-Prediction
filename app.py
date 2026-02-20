import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Premium Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ------------------------------------------------
# CUSTOM CSS (Glassmorphism + Gradient)
# ------------------------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #141E30, #243B55);
}
.big-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: white;
}
.sub-text {
    text-align: center;
    font-size: 18px;
    color: #dddddd;
}
.prediction-card {
    padding: 30px;
    border-radius: 15px;
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    color: white;
    font-size: 28px;
    text-align: center;
}
footer {
    text-align: center;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
model = joblib.load("car_price_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# ------------------------------------------------
# HEADER
# ------------------------------------------------
st.markdown("<div class='big-title'>🚗 Premium Car Price Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>An End-to-End Machine Learning Deployment Project</div>", unsafe_allow_html=True)
st.write("")

# ------------------------------------------------
# SIDEBAR CONTROLS
# ------------------------------------------------
st.sidebar.header("🔧 Configure Car Details")

year = st.sidebar.number_input("Year", 1990, 2024)
engine_size = st.sidebar.number_input("Engine Size", min_value=0.0)
mileage = st.sidebar.number_input("Mileage", min_value=0)

brand = st.sidebar.selectbox("Brand", ["Tesla", "BMW", "Audi", "Ford", "Toyota"])
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
condition = st.sidebar.selectbox("Condition", ["New", "Used", "Like New"])
model_name = st.sidebar.text_input("Model Name (Example: Mustang)")

predict_button = st.sidebar.button("🚀 Predict Price")

# ------------------------------------------------
# MAIN TABS
# ------------------------------------------------
tab1, tab2 = st.tabs(["💰 Prediction", "📊 Model Insights"])

# ------------------------------------------------
# PREDICTION TAB
# ------------------------------------------------
with tab1:

    if predict_button:

        input_dict = {
            "Year": year,
            "Engine Size": engine_size,
            "Mileage": mileage
        }

        input_df = pd.DataFrame([input_dict])

        categorical_inputs = {
            f"Brand_{brand}": 1,
            f"Fuel Type_{fuel_type}": 1,
            f"Transmission_{transmission}": 1,
            f"Condition_{condition}": 1,
            f"Model_{model_name}": 1
        }

        for col in categorical_inputs:
            input_df[col] = categorical_inputs[col]

        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        input_df = input_df[model_columns]

        prediction = model.predict(input_df)

        st.markdown(
            f"<div class='prediction-card'>Estimated Car Price: $ {prediction[0]:,.2f}</div>",
            unsafe_allow_html=True
        )

        st.success("Prediction Generated Successfully!")

    else:
        st.info("Enter car details in the sidebar and click Predict Price.")

# ------------------------------------------------
# MODEL INSIGHTS TAB
# ------------------------------------------------
with tab2:

    st.subheader("📊 Feature Importance")

    try:
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": model_columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        fig = px.bar(
            importance_df.head(10),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Important Features",
            color="Importance",
            color_continuous_scale="Blues"
        )

        st.plotly_chart(fig, use_container_width=True)

    except:
        st.warning("Feature importance not available for this model.")


