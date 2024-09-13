import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import pickle
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

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
st.markdown("<h1 style='text-align: center; color: #FF6347;'>Lung and Colon Cancer Prediction</h1>", unsafe_allow_html=True)

# Description of the application
st.markdown("""
<div style='border: 2px solid #2980B9; padding: 15px; border-radius: 8px; margin-top: 20px;'>
    <p style='color:#F0FFFF;'>On this page, you can predict the likelihood of Lung and Colon Cancer by uploading an image. Utilizing advanced Machine Learning models, our platform analyzes the uploaded image to provide insights into potential cancer indicators. Please remember that this tool is for educational purposes only and not a substitute for professional medical advice.</p>
</div>
""", unsafe_allow_html=True)

st.write("")
st.write("")

# Function to preprocess images
def preprocess_image(image, target_size=(256, 256)):
    image = np.array(image)
    image = image[:, :, :3]  
    image = tf.image.resize(image, target_size) 
    image = image / 255.0  
    image = tf.expand_dims(image, axis=0) 
    return image

# Load the model
model = tf.keras.models.load_model('.\\lung_colon_cancer\\LungColonCancer_model.h5')

# Class names and explanations
CLASS_NAMES = ['Colon Adenocarcinoma', 'Colon Benign Tissue', 'Lung Adenocarcinoma', 'Lung Benign Tissue', 'Lung Squamous Cell Carcinoma']

EXPLANATIONS = {
    'Colon Adenocarcinoma': "Colon Adenocarcinoma: The uploaded image indicates the presence of colon adenocarcinoma, a type of cancer that starts in the mucus-producing gland cells of the colon. Immediate medical consultation is recommended.",
    'Colon Benign Tissue': "Colon Benign Tissue: The uploaded image shows benign tissue in the colon, which is non-cancerous. However, regular monitoring and consultation with a healthcare professional are advised.",
    'Lung Adenocarcinoma': "Lung Adenocarcinoma: The uploaded image suggests lung adenocarcinoma, a common type of lung cancer that starts in the glandular cells. Early diagnosis and treatment are crucial.",
    'Lung Benign Tissue': "Lung Benign Tissue: The uploaded image shows benign lung tissue, which is non-cancerous. Regular check-ups and monitoring are recommended.",
    'Lung Squamous Cell Carcinoma': "Lung Squamous Cell Carcinoma: The uploaded image indicates lung squamous cell carcinoma, a type of lung cancer that starts in the squamous cells. Immediate medical consultation is recommended."
}

main_tabs = st.tabs(["Prediction", "Help"])

# Prediction tab
with main_tabs[0]:
    sub_tabs = st.tabs(["Image Prediction", "Value Prediction"])
    
    with sub_tabs[0]:
        uploaded_images = st.file_uploader(
            "Please upload one or more image files (jpg, jpeg, png, gif, bmp, tiff). Ensure the images are clear for accurate predictions.",
            type=["jpg", "jpeg", "png", "gif", "bmp", "tiff"], 
            accept_multiple_files=True
        )

        # Process the uploaded images
        if uploaded_images:
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
                    predictions = model.predict(processed_image)
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
                    
    with sub_tabs[1]:
        st.write("")
        
        # Columns to structure the layout
        col1, col2 = st.columns([3, 1])
        
        with col2:
            with st.expander("***Lung Cancer Data Measurements***", expanded=True):
                
                input_dict = {}
                
                # Input for Gender
                input_dict["gender"] = st.radio(
                    label="***Select Gender***",
                    options=["Male", "Female"],
                    help="Select your gender for lung cancer prediction.",
                    horizontal=True
                )
                
                # Input for Age
                input_dict["age"] = st.number_input(
                    label="***Enter Age***",
                    min_value=1,
                    max_value=120,
                    help="Enter your age."
                )

                # Input for Smoking
                input_dict["smoking"] = st.radio(
                    label="***Do you smoke?***",
                    options=["No", "Yes"],
                    help="Select if you smoke.",
                    horizontal=True
                )
                
                # Input for Yellow fingers
                input_dict["yellow_fingers"] = st.radio(
                    label="***Do you have yellow fingers?***",
                    options=["No", "Yes"],
                    help="Select if you have yellow fingers.",
                    horizontal=True
                )

                # Input for Anxiety
                input_dict["anxiety"] = st.radio(
                    label="***Do you experience anxiety?***",
                    options=["No", "Yes"],
                    help="Select if you experience anxiety.",
                    horizontal=True
                )

                # Input for Peer pressure
                input_dict["peer_pressure"] = st.radio(
                    label="***Do you experience peer pressure?***",
                    options=["No", "Yes"],
                    help="Select if you experience peer pressure.",
                    horizontal=True
                )

                # Input for Chronic Disease
                input_dict["chronic_disease"] = st.radio(
                    label="***Do you have a chronic disease?***",
                    options=["No", "Yes"],
                    help="Select if you have a chronic disease.",
                    horizontal=True
                )

                # Input for Fatigue
                input_dict["fatigue"] = st.radio(
                    label="***Do you experience fatigue?***",
                    options=["No", "Yes"],
                    help="Select if you experience fatigue.",
                    horizontal=True
                )

                # Input for Allergy
                input_dict["allergy"] = st.radio(
                    label="***Do you have any allergies?***",
                    options=["No", "Yes"],
                    help="Select if you have any allergies.",
                    horizontal=True
                )

                # Input for Wheezing
                input_dict["wheezing"] = st.radio(
                    label="***Do you experience wheezing?***",
                    options=["No", "Yes"],
                    help="Select if you experience wheezing.",
                    horizontal=True
                )

                # Input for Alcohol
                input_dict["alcohol"] = st.radio(
                    label="***Do you consume alcohol?***",
                    options=["No", "Yes"],
                    help="Select if you consume alcohol.",
                    horizontal=True
                )

                # Input for Coughing
                input_dict["coughing"] = st.radio(
                    label="***Do you experience coughing?***",
                    options=["No", "Yes"],
                    help="Select if you experience coughing.",
                    horizontal=True
                )

                # Input for Shortness of Breath
                input_dict["shortness_of_breath"] = st.radio(
                    label="***Do you experience shortness of breath?***",
                    options=["No", "Yes"],
                    help="Select if you experience shortness of breath.",
                    horizontal=True
                )

                # Input for Swallowing Difficulty
                input_dict["swallowing_difficulty"] = st.radio(
                    label="***Do you experience swallowing difficulty?***",
                    options=["No", "Yes"],
                    help="Select if you experience swallowing difficulty.",
                    horizontal=True
                )

                # Input for Chest pain
                input_dict["chest_pain"] = st.radio(
                    label="***Do you experience chest pain?***",
                    options=["No", "Yes"],
                    help="Select if you experience chest pain.",
                    horizontal=True
                )

        # Function to convert input data into a format suitable for the model
        def convert_inputs(input_data):
            # Convert yes/no answers to numerical values
            converted_data = {
                'gender': 0 if input_data["gender"] == "Male" else 1,
                'age': input_data["age"],
                'smoking': 2 if input_data["smoking"] == "Yes" else 1,
                'yellow_fingers': 2 if input_data["yellow_fingers"] == "Yes" else 1,
                'anxiety': 2 if input_data["anxiety"] == "Yes" else 1,
                'peer_pressure': 2 if input_data["peer_pressure"] == "Yes" else 1,
                'chronic_disease': 2 if input_data["chronic_disease"] == "Yes" else 1,
                'fatigue': 2 if input_data["fatigue"] == "Yes" else 1,
                'allergy': 2 if input_data["allergy"] == "Yes" else 1,
                'wheezing': 2 if input_data["wheezing"] == "Yes" else 1,
                'alcohol': 2 if input_data["alcohol"] == "Yes" else 1,
                'coughing': 2 if input_data["coughing"] == "Yes" else 1,
                'shortness_of_breath': 2 if input_data["shortness_of_breath"] == "Yes" else 1,
                'swallowing_difficulty': 2 if input_data["swallowing_difficulty"] == "Yes" else 1,
                'chest_pain': 2 if input_data["chest_pain"] == "Yes" else 1
            }
            return list(converted_data.values())

        with col1:
            if st.button('Predict Lung Cancer'):
                
                user_input = np.array(convert_inputs(input_dict)).reshape(1, -1)
                
                # Load the scaler
                scaler = StandardScaler()
                scaler = scaler.fit_transform(user_input)

                # Load the prediction model
                with open('.\\lung_colon_cancer\\lungcsvmodel.pkl', 'rb') as file:
                    prediction_model = pickle.load(file)

                # Make a prediction
                prediction = prediction_model.predict(scaler)
                probability = prediction_model.predict_proba(scaler)[0][int(prediction[0])]
                
                # Add this function after the predict button is clicked
                def plot_gauge_chart(probability):
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,
                        title={'text': "Lung Cancer Probability (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "red" if probability > 0.5 else "green"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgreen"},
                                {'range': [50, 100], 'color': "lightcoral"}],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': probability * 100}}))

                    st.plotly_chart(fig)

                # Assuming `probability` is calculated from the model
                plot_gauge_chart(probability)


                # Display the prediction result
                if prediction[0] == 1:
                    st.markdown("""
                    <div style='border: 2px solid #2980B9; padding: 15px; border-radius: 10px; margin-top: 20px; background-color: #FFDAB9;'>
                        <h2 style='text-align: center; color: #8B0000;'>Warning: Lung Cancer Detected</h2>
                        <p style='text-align: center; font-size: 18px;'>The model predicts that you might have lung cancer. The probability of this prediction is <strong>{:.2f}</strong>. It is strongly advised to consult a healthcare professional for further evaluation.</p>
                    </div>
                    """.format(probability), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='border: 2px solid #2980B9; padding: 15px; border-radius: 10px; margin-top: 20px; background-color: #90EE90;'>
                        <h2 style='text-align: center; color: #006400;'>Result: No Lung Cancer Detected</h2>
                        <p style='text-align: center; font-size: 18px;'>The model predicts that you do not have lung cancer. The probability of this prediction is <strong>{:.2f}</strong>. However, it is still advisable to maintain regular check-ups.</p>
                    </div>
                    """.format(probability), unsafe_allow_html=True)

with main_tabs[1]:

    st.markdown(f"""
        <div style='border: 2px solid #2980B9; padding: 9px; border-radius: 10px; margin-top: 10px;'>
        <h2 style='color: #FFFAFA;'>What Images to Upload</h2>
        <p style='color: #F0FFFF;'>To ensure accurate predictions, please follow these guidelines when uploading images:</p>
        <ul style='color: #F0FFFF;'>
            <li>Ensure that the images are clear and focused, allowing for accurate analysis of the details.</li>
            <li>Preferably upload medical images such as CT scans, MRI images, or histopathological images relevant to lung and colon cancers.</li>
            <li>The images should be well-lit, avoiding shadows or glare that may obscure important features.</li>
            <li>Use images of high resolution to ensure that the model can effectively analyze fine details. Images should ideally be at least 256x256 pixels.</li>
            <li>Avoid images that contain text, labels, or other obstructions that could interfere with the model's ability to identify cancer indicators.</li>
        </ul>
            <p style='color: #F0FFFF;'>If you have any further questions, please consult a medical professional.</p>
        </div>
        
        <div style='border: 2px solid #2980B9; padding: 15px; border-radius: 10px; margin-top: 20px;'>
            <h2 style='color: #FFFAFA;'>Understanding the Prediction Classes</h2>
            <ul style='color: #F0FFFF;'>
                <li><strong>Colon Adenocarcinoma:</strong> The uploaded image indicates the presence of colon adenocarcinoma. Early medical intervention is essential.</li>
                <li><strong>Colon Benign Tissue:</strong> The uploaded image shows benign tissue in the colon, which is non-cancerous. Monitoring and consultation are advised.</li>
                <li><strong>Lung Adenocarcinoma:</strong> The uploaded image suggests lung adenocarcinoma. Early diagnosis and treatment are crucial.</li>
                <li><strong>Lung Benign Tissue:</strong> The uploaded image shows benign lung tissue, which is non-cancerous. Regular check-ups are recommended.</li>
                <li><strong>Lung Squamous Cell Carcinoma:</strong> The uploaded image indicates lung squamous cell carcinoma. Immediate medical consultation is recommended.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")

        # Example images showing types of acceptable uploads
    st.markdown("<h3 style='color: #F0FFFF;'>Examples of Acceptable Images</h3>", unsafe_allow_html=True)

        # Add images here with the correct paths
    st.image(".\\lung_colon_cancer\\lung_bnt_0006.jpg", use_column_width=True)
