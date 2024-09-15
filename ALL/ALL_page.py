import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

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
st.markdown("<h1 style='text-align: center; color: #FF6347;'>Acute Lymphoblastic Leukemia Prediction</h1>", unsafe_allow_html=True)

# Description of the application
st.markdown("""
<div style='border: 2px solid #2980B9; padding: 15px; border-radius: 8px; margin-top: 20px;'>
    <p style='color:#F0FFFF;'>On this page, you can predict the likelihood of Acute Lymphoblastic Leukemia by uploading an image. Utilizing advanced Machine Learning models, our platform analyzes the uploaded image to provide insights into potential cancer indicators. Please remember that this tool is for educational purposes only and not a substitute for professional medical advice.</p>
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

# Create tabs for the application
tabs = st.tabs(["Prediction", "Help"])

# Load the model
model = tf.keras.models.load_model('./ALL/ALL_model.h5')

CLASS_NAMES = ['Benign', 'Early', 'Pre', 'Pro']

EXPLANATIONS = {
    'Benign': "Benign: Refers to a non-cancerous condition. In the context of ALL prediction, a 'benign' result would indicate that the cells or tissues analyzed are not showing signs of leukemia or any malignant behavior.",
    'Early': "Early: This typically refers to the initial stages of leukemia. In ALL, detecting the disease at an 'early' stage can be crucial for treatment and prognosis. Early-stage ALL might mean the presence of a smaller number of cancerous cells that have just begun to proliferate.",
    'Pre': "Pre: This might refer to a 'pre-leukemic' condition or the stage before full-blown leukemia develops. In ALL, a 'pre' condition could indicate abnormal changes in the blood cells that suggest an increased risk of developing leukemia but not yet a confirmed diagnosis.",
    'Pro': "Pro: This could refer to 'pro-leukemic' conditions, which are advanced and more aggressive stages of the disease. It could also indicate cells that are already exhibiting malignant properties, suggesting a more developed or advanced state of ALL."
    }

# Prediction tab
with tabs[0]:
    # File uploader for image upload
    uploaded_images = st.file_uploader(
        "Please upload one or more image files (jpg, jpeg, png, gif, bmp, tiff). Ensure the images are clear for accurate predictions.",
        type=["jpg", "jpeg", "png", "gif", "bmp", "tiff"], 
        accept_multiple_files=True
    )

    # Process the uploaded images
    if uploaded_images is not None:
        cols = st.columns(3)  

        # Iterate over the uploaded images
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

# Help tab
with tabs[1]:
    st.markdown(f"""
    <div style='border: 2px solid #2980B9; padding: 9px; border-radius: 10px; margin-top: 10px;'>
        <h2 style='color: #FFFAFA;'>What Images to Upload</h2>
        <p style='color: #F0FFFF;'>To ensure accurate predictions for Acute Lymphoblastic Leukemia, please follow these guidelines when uploading images:</p>
        <ul style='color: #F0FFFF;'>
            <li>Ensure that the images are clear and focused, allowing for accurate analysis of the details.</li>
            <li>Preferably upload medical images related to Acute Lymphoblastic Leukemia, such as blood smears showing leukemic cells or biopsy samples highlighting characteristics of ALL.</li>
            <li>The images should be well-lit, avoiding shadows or glare that may obscure important features.</li>
            <li>Use images of high resolution to ensure that the model can effectively analyze fine details. Images should ideally be at least 256x256 pixels.</li>
            <li>Avoid images that contain text, labels, or other obstructions that could interfere with the model's ability to identify cancer indicators.</li>
        </ul>
        <p style='color: #F0FFFF;'>If you have any further questions, please consult a medical professional.</p>
    </div>
    
    <div style='border: 2px solid #2980B9; padding: 15px; border-radius: 10px; margin-top: 20px;'>
        <h2 style='color: #FFFAFA;'>Understanding the Prediction Classes</h2>
        <ul style='color: #F0FFFF;'>
            <li><strong>Benign:</strong> Refers to a non-cancerous condition. In the context of ALL prediction, a "benign" result would indicate that the cells or tissues analyzed are not showing signs of leukemia or any malignant behavior.</li>
            <li><strong>Early:</strong> This typically refers to the initial stages of leukemia. In ALL, detecting the disease at an "early" stage can be crucial for treatment and prognosis. Early-stage ALL might mean the presence of a smaller number of cancerous cells that have just begun to proliferate.</li>
            <li><strong>Pre:</strong> This might refer to a "pre-leukemic" condition or the stage before full-blown leukemia develops. In ALL, a "pre" condition could indicate abnormal changes in the blood cells that suggest an increased risk of developing leukemia but not yet a confirmed diagnosis.</li>
            <li><strong>Pro:</strong> This could refer to "pro-leukemic" conditions, which are advanced and more aggressive stages of the disease. It could also indicate cells that are already exhibiting malignant properties, suggesting a more developed or advanced state of ALL.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.write("")

    # Example images showing types of acceptable uploads
    st.markdown("<h3 style='color: #F0FFFF;'>Examples of Acceptable Images</h3>", unsafe_allow_html=True)

    # Add images here with the correct paths
    st.image("./ALL/all_pro.jpg", use_column_width=True)
 

