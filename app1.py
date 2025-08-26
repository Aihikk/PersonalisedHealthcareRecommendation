import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
try:
    with open('logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('fit_characteristics.pkl', 'rb') as f:
        fit_characteristics = pickle.load(f)

except FileNotFoundError:
    st.error("Model, scaler, or fit characteristics file not found. Please train the model and save the files.")
    st.stop()


# Function to preprocess user input
def preprocess_input(input_data):
    # Create a DataFrame from input data
    input_df = pd.DataFrame([input_data])

    # Calculate BMI
    input_df['BMI'] = input_df['weight_kg'] / (input_df['height_cm'] / 100)**2

    # One-hot encode categorical variables (ensure all possible categories are considered)
    input_df = pd.get_dummies(input_df, columns=['smokes', 'gender'], drop_first=True)

    # Ensure all columns present during training are present in the input DataFrame
    # and in the correct order. Fill missing columns with 0.
    # This assumes the order of columns in X_train is consistent.
    # A more robust approach would save the column order during training.
    train_cols = scaler.feature_names_in_
    for col in train_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[train_cols]


    # Scale numerical features
    numerical_cols = ['age', 'height_cm', 'weight_kg', 'heart_rate', 'blood_pressure', 'sleep_hours', 'nutrition_quality', 'activity_index', 'BMI']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    return input_df

# Function to generate recommendations
def generate_recommendations(individual_data, fit_characteristics, model):
    recommendations = []

    # Define thresholds (these can be adjusted)
    activity_threshold_low = fit_characteristics['activity_index'] - 0.8
    activity_threshold_moderate = fit_characteristics['activity_index'] - 0.3
    nutrition_threshold_low = fit_characteristics['nutrition_quality'] - 0.5
    bmi_threshold_high = fit_characteristics['BMI'] + 1.0
    sleep_threshold_low = fit_characteristics['sleep_hours'] - 0.5

    # Get prediction probability
    # The model predicts on a single instance, so predict_proba returns a 2D array with one row
    fit_probability = model.predict_proba(preprocess_input(individual_data))[:, 1][0]


    # Exercise Recommendation (incorporating age and activity level)
    # Need to get the original age before scaling for comparison
    original_age = individual_data['age']
    average_age = fit_characteristics['age'] # Using scaled average age for comparison with scaled input age
    if individual_data['activity_index'] < activity_threshold_low:
         # Using original age for more interpretable comparison
         if original_age < 40: # Example age threshold
              recommendations.append("Recommendation: Significantly increase your physical activity. Consider consulting a trainer for a vigorous exercise plan.")
         else:
               recommendations.append("Recommendation: Substantially increase moderate-intensity physical activity like brisk walking, swimming, or cycling.")
    elif individual_data['activity_index'] < activity_threshold_moderate:
         recommendations.append("Recommendation: Increase your physical activity level. Aim for more frequent or longer workouts.")


    # Dietary Recommendations (incorporating nutrition quality, BMI, and combined factors)
    if individual_data['nutrition_quality'] < nutrition_threshold_low and individual_data['BMI'] > bmi_threshold_high:
        recommendations.append("Recommendation: Focus on a comprehensive diet and weight management plan. Consult a nutritionist for personalized guidance.")
    elif individual_data['nutrition_quality'] < nutrition_threshold_low:
        recommendations.append("Recommendation: Improve the quality of your diet by incorporating more nutrient-dense foods like fruits, vegetables, and whole grains.")
    elif individual_data['BMI'] > bmi_threshold_high:
         recommendations.append("Recommendation: Focus on portion control and making healthier food choices to manage your weight.")


    # Sleep Recommendation (incorporating sleep hours and probability)
    if individual_data['sleep_hours'] < sleep_threshold_low and fit_probability < 0.5: # Example: Stronger recommendation for lower probability
         recommendations.append("Recommendation: Prioritize sleep hygiene and aim for 7-9 hours of quality sleep. Consider consulting a sleep specialist if needed.")
    elif individual_data['sleep_hours'] < sleep_threshold_low:
        recommendations.append("Recommendation: Aim for more consistent and sufficient sleep. Practice good sleep hygiene.")


    # Smoking Recommendation
    if individual_data['smokes'] == 'yes' or individual_data['smokes'] == '1':
        recommendations.append("Recommendation: Quitting smoking is crucial for improving your fitness and overall health. Seek support if needed.")


    return recommendations, fit_probability


# Streamlit App
st.title("Personalized Healthcare Recommendation System")

st.write("Enter your health data to get personalized fitness predictions and recommendations.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=30)
height_cm = st.number_input("Height (cm)", min_value=50.0, max_value=300.0, value=170.0)
weight_kg = st.number_input("Weight (kg)", min_value=10.0, max_value=500.0, value=70.0)
heart_rate = st.number_input("Heart Rate", min_value=30.0, max_value=200.0, value=70.0)
blood_pressure = st.number_input("Blood Pressure", min_value=50.0, max_value=300.0, value=120.0)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)
nutrition_quality = st.number_input("Nutrition Quality (1-10)", min_value=1.0, max_value=10.0, value=5.0)
activity_index = st.number_input("Activity Index (1-5)", min_value=1.0, max_value=5.0, value=3.0)
smokes = st.selectbox("Smokes", ['no', 'yes', '0', '1'])
gender = st.selectbox("Gender", ['F', 'M'])

# Create a dictionary with input data
input_data = {
    'age': age,
    'height_cm': height_cm,
    'weight_kg': weight_kg,
    'heart_rate': heart_rate,
    'blood_pressure': blood_pressure,
    'sleep_hours': sleep_hours,
    'nutrition_quality': nutrition_quality,
    'activity_index': activity_index,
    'smokes': smokes,
    'gender': gender
}

if st.button("Get Recommendations"):
    # Preprocess input and make prediction
    processed_input = preprocess_input(input_data)
    prediction = model.predict(processed_input)[0]
    prediction_proba = model.predict_proba(processed_input)[0][1] # Probability of being fit


    st.subheader("Fitness Prediction")
    if prediction == 1:
        st.success(f"Predicted Fitness Status: Fit (Probability: {prediction_proba:.2f})")
    else:
        st.warning(f"Predicted Fitness Status: Not Fit (Probability: {prediction_proba:.2f})")

    st.subheader("Personalized Recommendations")
    if prediction == 0: # Only provide recommendations if predicted as not fit
        # Need to pass the original input data to the recommendation function
        recommendations, fit_probability_rec = generate_recommendations(input_data, fit_characteristics, model)
        if recommendations:
            for rec in recommendations:
                st.write(f"- {rec}")
        else:
            st.write("No specific recommendations based on current rules for this profile.")
    else:
        st.info("Great job! You are predicted as fit based on your data. Keep up the healthy habits!")
