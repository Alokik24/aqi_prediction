import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

# Load saved components
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)
with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

# Streamlit UI
st.title("Air Quality Index (AQI) Prediction App")

# Manual Input for AQI Prediction
st.header("Manual Input")
st.markdown("Provide feature values to predict AQI.")

# Input sliders for numerical features
dewp = st.slider("Dew Point (DEWP)", min_value=-30.0, max_value=30.0, value=0.0)
temp = st.slider("Temperature (TEMP)", min_value=-20.0, max_value=50.0, value=20.0)
pres = st.slider("Pressure (PRES)", min_value=950.0, max_value=1050.0, value=1010.0)
iws = st.slider("Cumulated Wind Speed (Iws)", min_value=0.0, max_value=50.0, value=10.0)
day_of_week = st.slider("Day of the Week (0=Mon, 6=Sun)", min_value=0, max_value=6, value=0)
month_of_year = st.slider("Month of the Year", min_value=1, max_value=12, value=1)

# Inputs for one-hot encoded wind direction
st.markdown("**Wind Direction (cbwd)**")
cbwd_NE = st.checkbox("NE")
cbwd_NW = st.checkbox("NW")
cbwd_SE = st.checkbox("SE")

# Prepare input for prediction
input_data = [dewp, temp, pres, iws, day_of_week, month_of_year, int(cbwd_NE), int(cbwd_NW), int(cbwd_SE)]

if st.button("Predict AQI (Manual Input)"):
    # Preprocess input data
    input_scaled = scaler.transform([input_data])
    input_pca = pca.transform(input_scaled)
    predicted_aqi = best_model.predict(input_pca)[0]
    st.success(f"Predicted AQI: {predicted_aqi:.2f}")

# Dataset Upload for AQI Prediction
st.header("Upload Dataset")
st.markdown("Upload a dataset for batch AQI prediction.")

data_file = st.file_uploader("Upload your CSV file", type="csv")

if data_file:
    # Load and display the dataset
    df = pd.read_csv(data_file)
    st.write("Preview of the dataset", df.head())

    # Preprocessing the data
    df = df.replace('NA', np.nan)
    numeric_columns = ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    # Generate AQI values
    def calculate_aqi(pm25):
        breakpoints = [
            (0, 12.0, 0, 50),
            (12.1, 35.4, 51, 100),
            (35.5, 55.4, 101, 150),
            (55.5, 150.4, 151, 200),
            (150.5, 250.4, 201, 300),
            (250.5, 350.4, 301, 400),
            (350.5, 500.4, 401, 500)
        ]
        for low_conc, high_conc, low_aqi, high_aqi in breakpoints:
            if low_conc <= pm25 <= high_conc:
                return ((high_aqi - low_aqi) / (high_conc - low_conc)) * (pm25 - low_conc) + low_aqi
        return 500 if pm25 > 500.4 else None

    df['AQI'] = df['pm2.5'].apply(calculate_aqi)

    # Feature engineering
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month_of_year'] = df['date'].dt.month

    # One-hot encode categorical variables (e.g., wind direction)
    df = pd.get_dummies(df, columns=['cbwd'], drop_first=True)

    # Define features and target
    feature_columns = ['DEWP', 'TEMP', 'PRES', 'Iws', 'day_of_week', 'month_of_year']
    categorical_columns = [col for col in df.columns if col.startswith('cbwd_')]
    feature_columns.extend(categorical_columns)
    X = df[feature_columns]

    # Apply scaling and PCA
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    # Predict AQI using the loaded model
    predictions = best_model.predict(X_pca)

    # Display results
    df['Predicted_AQI'] = predictions
    st.write("Predicted AQI values for your data:", df[['date', 'AQI', 'Predicted_AQI']])

    # Visualize actual vs predicted AQI
    st.subheader("Actual vs Predicted AQI")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['AQI'], df['Predicted_AQI'], alpha=0.5)
    ax.plot([df['AQI'].min(), df['AQI'].max()], [df['AQI'].min(), df['AQI'].max()], 'r--', lw=2)
    ax.set_xlabel('Actual AQI')
    ax.set_ylabel('Predicted AQI')
    ax.set_title('Actual vs Predicted AQI')
    st.pyplot(fig)

    # Time-Series Analysis
    st.header("Time-Series Analysis")
    df = df.sort_values(by='date')
    st.line_chart(df[['date', 'Predicted_AQI']].set_index('date'))

    # Option to download the predicted results
    st.download_button(
        label="Download Predicted AQI CSV",
        data=df[['date', 'AQI', 'Predicted_AQI']].to_csv(index=False),
        file_name='predicted_aqi.csv',
        mime='text/csv'
    )
