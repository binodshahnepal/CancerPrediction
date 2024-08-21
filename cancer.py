import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier  # Replace with your model

# Sample dataset preparation (replace with your dataset)
data = {  # Example data; replace with actual data
    'radius_mean': np.random.rand(569),
    'texture_mean': np.random.rand(569),
    'perimeter_mean': np.random.rand(569),
    'area_mean': np.random.rand(569),
    'smoothness_mean': np.random.rand(569),
    'compactness_mean': np.random.rand(569),
    'concavity_mean': np.random.rand(569),
    'concave points_mean': np.random.rand(569),
    'symmetry_mean': np.random.rand(569),
    'fractal_dimension_mean': np.random.rand(569),
    'radius_se': np.random.rand(569),
    'texture_se': np.random.rand(569),
    'perimeter_se': np.random.rand(569),
    'area_se': np.random.rand(569),
    'smoothness_se': np.random.rand(569),
    'compactness_se': np.random.rand(569),
    'concavity_se': np.random.rand(569),
    'concave points_se': np.random.rand(569),
    'symmetry_se': np.random.rand(569),
    'fractal_dimension_se': np.random.rand(569),
    'radius_worst': np.random.rand(569),
    'texture_worst': np.random.rand(569),
    'perimeter_worst': np.random.rand(569),
    'area_worst': np.random.rand(569),
    'smoothness_worst': np.random.rand(569),
    'compactness_worst': np.random.rand(569),
    'concavity_worst': np.random.rand(569),
    'concave points_worst': np.random.rand(569),
    'symmetry_worst': np.random.rand(569),
    'fractal_dimension_worst': np.random.rand(569),
    'diagnosis': np.random.choice(['M', 'B'], 569)
}

df = pd.DataFrame(data)

# Features and target variable
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Label Encoding for target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Model training
model = AdaBoostClassifier(random_state=42)  # Replace with your model
model.fit(X_train, y_train)

# Save the model and LabelEncoder
joblib.dump(model, 'trained_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Streamlit Web Application
st.title('Breast Cancer Prediction')

# User inputs for all features
radius_mean = st.number_input('Radius Mean', min_value=0.0, max_value=100.0, value=10.0)
texture_mean = st.number_input('Texture Mean', min_value=0.0, max_value=100.0, value=10.0)
perimeter_mean = st.number_input('Perimeter Mean', min_value=0.0, max_value=500.0, value=100.0)
area_mean = st.number_input('Area Mean', min_value=0.0, max_value=2000.0, value=500.0)
smoothness_mean = st.number_input('Smoothness Mean', min_value=0.0, max_value=1.0, value=0.1)
compactness_mean = st.number_input('Compactness Mean', min_value=0.0, max_value=1.0, value=0.1)
concavity_mean = st.number_input('Concavity Mean', min_value=0.0, max_value=1.0, value=0.1)
concave_points_mean = st.number_input('Concave Points Mean', min_value=0.0, max_value=1.0, value=0.1)
symmetry_mean = st.number_input('Symmetry Mean', min_value=0.0, max_value=1.0, value=0.1)
fractal_dimension_mean = st.number_input('Fractal Dimension Mean', min_value=0.0, max_value=1.0, value=0.1)
radius_se = st.number_input('Radius SE', min_value=0.0, max_value=100.0, value=10.0)
texture_se = st.number_input('Texture SE', min_value=0.0, max_value=100.0, value=10.0)
perimeter_se = st.number_input('Perimeter SE', min_value=0.0, max_value=500.0, value=100.0)
area_se = st.number_input('Area SE', min_value=0.0, max_value=2000.0, value=500.0)
smoothness_se = st.number_input('Smoothness SE', min_value=0.0, max_value=1.0, value=0.1)
compactness_se = st.number_input('Compactness SE', min_value=0.0, max_value=1.0, value=0.1)
concavity_se = st.number_input('Concavity SE', min_value=0.0, max_value=1.0, value=0.1)
concave_points_se = st.number_input('Concave Points SE', min_value=0.0, max_value=1.0, value=0.1)
symmetry_se = st.number_input('Symmetry SE', min_value=0.0, max_value=1.0, value=0.1)
fractal_dimension_se = st.number_input('Fractal Dimension SE', min_value=0.0, max_value=1.0, value=0.1)
radius_worst = st.number_input('Radius Worst', min_value=0.0, max_value=100.0, value=10.0)
texture_worst = st.number_input('Texture Worst', min_value=0.0, max_value=100.0, value=10.0)
perimeter_worst = st.number_input('Perimeter Worst', min_value=0.0, max_value=500.0, value=100.0)
area_worst = st.number_input('Area Worst', min_value=0.0, max_value=2000.0, value=500.0)
smoothness_worst = st.number_input('Smoothness Worst', min_value=0.0, max_value=1.0, value=0.1)
compactness_worst = st.number_input('Compactness Worst', min_value=0.0, max_value=1.0, value=0.1)
concavity_worst = st.number_input('Concavity Worst', min_value=0.0, max_value=1.0, value=0.1)
concave_points_worst = st.number_input('Concave Points Worst', min_value=0.0, max_value=1.0, value=0.1)
symmetry_worst = st.number_input('Symmetry Worst', min_value=0.0, max_value=1.0, value=0.1)
fractal_dimension_worst = st.number_input('Fractal Dimension Worst', min_value=0.0, max_value=1.0, value=0.1)

# Predict button
if st.button('Predict'):
    # Load the saved model and LabelEncoder
    model = joblib.load('trained_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')

    # Prepare the feature vector for prediction
    features = np.array([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, 
                          compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, 
                          fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, 
                          smoothness_se, compactness_se, concavity_se, concave_points_se, 
                          symmetry_se, fractal_dimension_se, radius_worst, texture_worst, 
                          perimeter_worst, area_worst, smoothness_worst, compactness_worst, 
                          concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]])
    
    # Make the prediction
    prediction = model.predict(features)
    prediction_label = label_encoder.inverse_transform(prediction)[0]
    
    st.write(f'The predicted diagnosis is: {prediction_label}')

# Run the app with: streamlit run app.py
