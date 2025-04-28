import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the saved model and LabelEncoder (trained model and label encoder should already be saved before)
model = pickle.load(open('model.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

# Load dataset
df = pd.read_csv("Nadi_Pariksha_Dataset.csv")

# Streamlit Interface
st.title("Dosha Type Prediction and Recommendation")

# User inputs with unit for temperature
age = st.number_input("Enter Age", min_value=1, max_value=100, value=25)
weight = st.number_input("Enter Weight", min_value=1, max_value=200, value=70)
height = st.number_input("Enter Height", min_value=1, max_value=250, value=170)
bp = st.number_input("Enter Blood Pressure", min_value=1, max_value=200, value=120)
temperature = st.number_input("Enter Temperature (°C)", min_value=20.0, max_value=45.0, value=37.0)
pulse_rate = st.number_input("Enter Pulse Rate", min_value=1, max_value=200, value=72)

# Create a DataFrame for the new input sample
new_sample = np.array([[age, weight, height, bp, temperature, pulse_rate]])

# Button to trigger prediction
if st.button("Predict Dosha Type"):
    # Predict using the trained model
    prediction = model.predict(new_sample)
    predicted_dosha = le.inverse_transform(prediction)


    # Get the recommendation for the predicted Dosha type
    def give_recommendation(dosha_type):
        recommendations = {
            "Vata": "You should focus on staying warm and grounded. Include calming activities like meditation and yoga.",
            "Pitta": "Cooling foods and relaxation techniques can help you stay balanced. Avoid overheating and stress.",
            "Kapha": "Stimulation and movement are key for you. Engage in vigorous physical activities and avoid heavy foods.",
            "Vata-Pitta": "Ensure you stay balanced with a mix of calming and cooling practices. Yoga and cooling foods work well for you.",
            "Kapha-Pitta": "Balance your fiery nature with cooling, yet stimulating activities. Avoid heavy foods and stress.",
            "Vata-Kapha": "Balance your grounded nature with invigorating activities. Ensure warmth and grounding practices to stay centered."
        }
        return recommendations.get(dosha_type, "No recommendation available.")


    # Yoga Pose Recommendations
    def yoga_recommendation(dosha_type):
        yoga_poses = {
            "Vata": {
                "pose": "Tree Pose (Vrikshasana)",
                "image_url": "https://images.unsplash.com/photo-1534096210335-a3b961613bb5?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8dHJlZSUyMHBvc2V8ZW58MHx8MHx8fDA%3D"
                # Image URL from yoga journal
            },
            "Pitta": {
                "pose": "Child's Pose (Balasana)",
                "image_url": "https://images.unsplash.com/photo-1594928612032-c097872687f4?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8Q2hpbGQncyUyMFBvc2UlMjAoQmFsYXNhbmEpfGVufDB8fDB8fHww"
                # Replace with actual image URL
            },
            "Kapha": {
                "pose": "Sun Salutation (Surya Namaskar)",
                "image_url": "https://images.unsplash.com/photo-1606663368493-131f4f97c095?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8U3VyeWElMjBOYW1hc2thcnxlbnwwfHwwfHx8MA%3D%3D"
                # Replace with actual image URL
            },
            "Vata-Pitta": {
                "pose": "Seated Forward Bend (Paschimottanasana)",
                "image_url": "https://images.pexels.com/photos/3822191/pexels-photo-3822191.jpeg?auto=compress&cs=tinysrgb&w=600"
                # Replace with actual image URL
            },
            "Kapha-Pitta": {
                "pose": "Warrior Pose (Virabhadrasana)",
                "image_url": "https://plus.unsplash.com/premium_photo-1664053011458-922b8259e739?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8OXx8VmlyYWJoYWRyYXNhbmF8ZW58MHx8MHx8fDA%3D"
                # Replace with actual image URL
            },
            "Vata-Kapha": {
                "pose": "Child’s Pose (Balasana)",
                "image_url": "https://images.unsplash.com/photo-1594928612032-c097872687f4?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8Q2hpbGQncyUyMFBvc2UlMjAoQmFsYXNhbmEpfGVufDB8fDB8fHww"
                # Replace with actual image URL
            }
        }
        return yoga_poses.get(dosha_type, {"pose": "No yoga pose available", "image_url": ""})


    # Get the yoga pose for the predicted Dosha type
    yoga_pose_info = yoga_recommendation(predicted_dosha[0])

    recommendation = give_recommendation(predicted_dosha[0])

    # Display results with bold headings
    st.markdown(f"**Predicted Dosha Type:** {predicted_dosha[0]}")
    st.markdown(f"**Recommendation:** {recommendation}")
    st.markdown(f"**Recommended Yoga Pose:** {yoga_pose_info['pose']}")

    if yoga_pose_info["image_url"]:
        st.image(yoga_pose_info["image_url"], caption=f"{yoga_pose_info['pose']} Image", use_container_width=True)

    # Data Visualization
    st.subheader("Feature Importance")
    importance = model.feature_importances_
    features = df.columns[:-1]  # Exclude the target column
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    # Plot Feature Importance
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax, palette="viridis")
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    numerical_df = df.select_dtypes(include=np.number)
    sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    st.pyplot(fig)
