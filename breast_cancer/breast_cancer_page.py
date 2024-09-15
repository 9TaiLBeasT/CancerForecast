import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import pickle
import plotly.graph_objects as go


# Load external CSS
with open(r"styles/main.css") as f:
    st.write(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Add glowing lines and background
background = """
    <style>
    [data-testid="stSidebar"] > div:first-child {
        background-image: url("https://devsnap.nyc3.digitaloceanspaces.com/devsnap.me/codepen-VjrZWv.png");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    <div class="glowing-line"></div>
    <div class="glowing-line"></div>
    <div class="glowing-line"></div>
    <div class="glowing-line"></div>
    <div class="glowing-line"></div>
    <div class="glowing-line"></div>
    <div class="glowing-line"></div>
"""
st.markdown(background, unsafe_allow_html=True)

# Title of the application
st.markdown("<h1 style='text-align: center; color: #FF6347;'>Breast Cancer Prediction</h1>", unsafe_allow_html=True)

# Description of the application
st.markdown("""
<div style='border: 2px solid #2980B9; padding: 15px; border-radius: 8px; margin-top: 20px;'>
    <p style='color:#F0FFFF;'>On this page, you can predict the likelihood of Breast Cancer either by uploading an image or by entering specific medical values. Please remember that this tool is for educational purposes only and not a substitute for professional medical advice.</p>
</div>
""", unsafe_allow_html=True)

st.write("")
st.write("")

# Function to preprocess images
def preprocess_image(image, target_size=(256, 256)):
    image = np.array(image)
    image = image[:, :, :3]  # Ensure image has 3 color channels
    image = tf.image.resize(image, target_size)  # Resize image
    image = image / 255.0  # Normalize image
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Load the models
image_model = tf.keras.models.load_model('./breast_cancer/BreastCancer_model.h5')
value_model = pickle.load(open("./breast_cancer/Breast_csv_model.pkl", "rb"))
scaler = pickle.load(open("./scalar.pkl", "rb"))

# Class names and explanations for image prediction
CLASS_NAMES = ['Benign', 'Malignant']
EXPLANATIONS = {
    'Benign': "Benign: Indicates a non-cancerous tumor. These tumors do not spread to other parts of the body and are usually not life-threatening. However, they may still require treatment or monitoring.",
    'Malignant': "Malignant: Refers to a cancerous tumor. Malignant tumors can invade surrounding tissues and spread to other parts of the body, which is why early detection and treatment are critical."
}

# Create main tabs for the application
main_tabs = st.tabs(["Prediction", "Help"])

# Prediction tab with sub-tabs for Image Prediction and Value Prediction
with main_tabs[0]:
    sub_tabs = st.tabs(["Image Prediction", "Value Prediction"])

    # Image Prediction sub-tab
    with sub_tabs[0]:
        uploaded_images = st.file_uploader(
            "Please upload one or more image files (jpg, jpeg, png, gif, bmp, tiff). Ensure the images are clear for accurate predictions.",
            type=["jpg", "jpeg", "png", "gif", "bmp", "tiff"], 
            accept_multiple_files=True
        )

        # Process the uploaded images
        if uploaded_images is not None:
            cols = st.columns(3)  

            for idx, image in enumerate(uploaded_images):
                img = Image.open(image) 
                resized_img = img.resize((200, 200)) 
                
                col_idx = idx % 3  
                with cols[col_idx]:
                    st.image(resized_img, caption="Uploaded Image", use_column_width=False)

                    # Preprocess the image
                    processed_image = preprocess_image(resized_img)

                    # Predict the class
                    predictions = image_model.predict(processed_image)
                    predicted_class = np.argmax(predictions, axis=1)[0]
                    predicted_label = CLASS_NAMES[predicted_class]
                    probability = predictions[0][predicted_class]
                    
                    explanation = EXPLANATIONS[predicted_label]

                    # Display the prediction
                    prediction_html = f"""
                    <div style='border: 2px solid #2980B9; padding: 9px; border-radius: 10px; margin-top: 10px;'>
                        <h3 style='color: #2980B9;'>Prediction Result</h3>
                        <p style='font-weight: bold;'>
                            Predicted class: <span style='color: #33cc33;'>{predicted_label}</span><br>
                            Probability: <span style='color: #ff9900;'>{probability:.2f}</span>
                        </p>
                        <p style='color: #AFEEEE;'>{explanation}</p>
                    </div>
                    """
                    st.markdown(prediction_html, unsafe_allow_html=True)

    # Value Prediction sub-tab
    with sub_tabs[1]:
        st.write("")  # Adding space between components

        # Columns to structure the layout
        col1, col2 = st.columns([3, 1])

        with col2:  # Right-side column for sliders
            with st.expander("Cell Nuclei Measurements", expanded=True):
            
                slider_labels = [
                    ("Radius (mean)", "radius_mean"),
                    ("Texture (mean)", "texture_mean"),
                    ("Perimeter (mean)", "perimeter_mean"),
                    ("Area (mean)", "area_mean"),
                    ("Smoothness (mean)", "smoothness_mean"),
                    ("Compactness (mean)", "compactness_mean"),
                    ("Concavity (mean)", "concavity_mean"),
                    ("Concave points (mean)", "concave points_mean"),
                    ("Symmetry (mean)", "symmetry_mean"),
                    ("Fractal dimension (mean)", "fractal_dimension_mean"),
                    ("Radius (se)", "radius_se"),
                    ("Texture (se)", "texture_se"),
                    ("Perimeter (se)", "perimeter_se"),
                    ("Area (se)", "area_se"),
                    ("Smoothness (se)", "smoothness_se"),
                    ("Compactness (se)", "compactness_se"),
                    ("Concavity (se)", "concavity_se"),
                    ("Concave points (se)", "concave points_se"),
                    ("Symmetry (se)", "symmetry_se"),
                    ("Fractal dimension (se)", "fractal_dimension_se"),
                    ("Radius (worst)", "radius_worst"),
                    ("Texture (worst)", "texture_worst"),
                    ("Perimeter (worst)", "perimeter_worst"),
                    ("Area (worst)", "area_worst"),
                    ("Smoothness (worst)", "smoothness_worst"),
                    ("Compactness (worst)", "compactness_worst"),
                    ("Concavity (worst)", "concavity_worst"),
                    ("Concave points (worst)", "concave points_worst"),
                    ("Symmetry (worst)", "symmetry_worst"),
                    ("Fractal dimension (worst)", "fractal_dimension_worst"),
                ]
                
                input_dict = {}
                
                for label, key in slider_labels:
                    input_dict[key] = st.slider(
                        label, 
                        min_value=0.0,  # Set as float
                        max_value=100.0,  # Set as float
                        value=50.0  # Set as float
                    )

            def get_scaled_values(input_dict):
                # Assume all values are between 0 and 1 for simplicity
                scaled_dict = {key: value / 100 for key, value in input_dict.items()}
                return scaled_dict

            def get_radar_chart(input_data):
                input_data = get_scaled_values(input_data)
                categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                            'Smoothness', 'Compactness', 
                            'Concavity', 'Concave Points',
                            'Symmetry', 'Fractal Dimension']

                fig = go.Figure()

                fig.add_trace(go.Scatterpolar(
                    r=[
                        input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
                        input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
                        input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
                        input_data['fractal_dimension_mean']
                    ],
                    theta=categories,
                    fill='toself',
                    name='Mean Value'
                ))
                fig.add_trace(go.Scatterpolar(
                    r=[
                        input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
                        input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
                        input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
                    ],
                    theta=categories,
                    fill='toself',
                    name='Standard Error'
                ))
                fig.add_trace(go.Scatterpolar(
                    r=[
                        input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
                        input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
                        input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
                        input_data['fractal_dimension_worst']
                    ],
                    theta=categories,
                    fill='toself',
                    name='Worst Value'
                ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True
                )

                return fig

            def add_predictions(input_data):
                input_array = np.array(list(input_data.values())).reshape(1, -1)
                input_array_scaled = scaler.transform(input_array)

                with col1:  # Left-side column for displaying predictions and charts
                    try:
                        prediction = value_model.predict(input_array_scaled)[0]
                        probabilities = value_model.predict_proba(input_array_scaled)[0]
                        prediction_label = "Malignant" if prediction == 1 else "Benign"
                        probability = probabilities[1] if prediction == 1 else probabilities[0]
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                        prediction_label = "Error"
                        probability = 0.0
                        
                    st.success(f"The prediction based on the input values is: {prediction_label}")
                    st.info(f"Model Confidence: {probability:.2f} ({prediction_label})")

                    # Display the radar chart
                    radar_chart = get_radar_chart(input_dict)
                    st.plotly_chart(radar_chart)

            with col2:
                if st.button("Predict", key="value_prediction"):
                    add_predictions(input_dict)

# Help tab for general guidance
with main_tabs[1]:
    st.header("Help and Documentation")

    st.markdown("""
    - **Image Prediction**: Upload a clear image for accurate predictions. You can upload multiple images.
    - **Value Prediction**: Use the sliders to input medical values related to Breast Cancer, then click 'Predict' to see the result.
    - **Disclaimer**: This tool is for educational purposes only. Always consult a medical professional for serious health concerns.
    """)

    st.markdown(f"""
    <div style='border: 2px solid #2980B9; padding: 15px; border-radius: 10px; margin-top: 20px;'>
        <h2 style='color: #FFFAFA;'>Understanding the Prediction Classes</h2>
        <ul style='color: #F0FFFF;'>
            <li><strong>Benign:</strong> Indicates a non-cancerous tumor. These tumors do not spread to other parts of the body and are usually not life-threatening. However, they may still require treatment or monitoring.</li>
            <li><strong>Malignant:</strong> Refers to a cancerous tumor. Malignant tumors can invade surrounding tissues and spread to other parts of the body, which is why early detection and treatment are critical.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    st.markdown("<h3 style='color: #F0FFFF;'>Examples of Acceptable Images</h3>", unsafe_allow_html=True)

    st.image("./breast_cancer/breast_cancer.jpg", use_column_width=True)
