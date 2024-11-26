# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ttest_ind
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture
import pickle

# Step 1: Data Collection and Preprocessing
# Load data
data_path = 'PRSA_data_2010.1.1-2014.12.31.csv'  # Path to uploaded file
df = pd.read_csv(data_path)

# Handle missing values and data types
df = df.replace('NA', np.nan)
numeric_columns = ['pm2.5', 'DEWP', 'TEMP', 'PRES', 'Iws', 'Is', 'Ir']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
df = df.dropna()

# Generate AQI values based on US EPA standards
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

# Step 2: Exploratory Data Analysis (EDA)
# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df[numeric_columns + ['AQI']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Pairplot for understanding feature relationships
sns.pairplot(df[numeric_columns + ['AQI']])
plt.show()

# Step 3: Feature Engineering
# Extract time-based features
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
y = df['AQI']

# Step 4: Dimensionality Reduction (PCA)
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to retain 95% variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Step 5: Statistical Validation
# Perform t-tests to validate feature significance (simple approach)
stat_tests = []
for col in X.columns:
    t_stat, p_val = ttest_ind(X[col], y)
    stat_tests.append((col, p_val))

# Display statistically significant features
significant_features = [col for col, p_val in stat_tests if p_val < 0.05]
print(f"Statistically significant features: {significant_features}")

# Step 6: Model Training and Evaluation
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Initialize models for comparison
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
}

# Train and evaluate models
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results.append((name, mae, rmse, r2))
    print(f"{name}: MAE = {mae}, RMSE = {rmse}, R² = {r2}")

# Step 7: Visualizations
# Compare predicted vs actual AQI for the best model (highest R² score)
best_model_name = max(results, key=lambda x: x[3])[0]
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)

# Scatter plot for predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual AQI')
plt.ylabel('Predicted AQI')
plt.title(f'Actual vs Predicted AQI ({best_model_name})')
plt.show()

# Step 8: Gaussian Mixture Model (Clustering)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)

# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=gmm_labels, cmap='viridis')
plt.title('Clustering with Gaussian Mixture Model')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Step 9: Build a Pipeline and Save Predictions
# Create a pipeline for preprocessing and model prediction
pipeline = Pipeline([ 
    ('scaler', StandardScaler()), 
    ('pca', PCA(n_components=0.95)), 
    ('model', best_model) 
])

# Fit the pipeline on the entire dataset
pipeline.fit(X, y)

# Predict AQI values for the dataset
predictions = pipeline.predict(X)

# Save the predictions to a CSV file
output_df = df.copy()
output_df['Predicted_AQI'] = predictions
output_df[['date', 'AQI', 'Predicted_AQI']].to_csv('predicted_aqi.csv', index=False)
print("Predicted AQI values saved to 'predicted_aqi.csv'")

# Step 10: Save the Models and Relevant Files
# Save the trained models
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
    
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
with open('pca.pkl', 'wb') as f:
    pickle.dump(pca, f)
    
# Optionally, you can also save the entire pipeline for easier use
with open('pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Models and relevant files saved.")
