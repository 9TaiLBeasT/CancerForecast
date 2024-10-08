# CancerForecast

## Project Overview

This project is an AI-powered cancer detection system developed to predict the likelihood of various types of cancer using machine learning models and medical images. The system currently includes models for Lymphoma Cancer, Brain Cancer, Acute Lymphoblastic Leukemia, Breast Cancer, Cervical Cancer, Kidney Cancer, and Lung and Colon Cancer.

The application provides a user-friendly interface where users can upload medical images, and the system will analyze the images to predict the likelihood of specific cancers. The predictions are based on advanced machine learning models trained on relevant datasets.

## Features

- **Image Upload:** Users can upload medical images in various formats (JPG, JPEG, PNG, GIF, BMP, TIFF).
- **Cancer Prediction:** The system predicts the likelihood of different cancers and provides explanations based on the predictions.
- **Interactive UI:** Includes tabs for Prediction and Help, making it easy to navigate and understand the system.
- **Visual Effects:** The application includes visual enhancements such as glowing lines and background effects to improve user experience.
- **Help Section:** Provides guidance on acceptable image types and explanations of prediction classes.

## Technologies Used

- **Streamlit:** Framework for creating the web application.
- **TensorFlow:** Machine learning library used to build and deploy models.
- **PIL (Python Imaging Library):** Library for image processing.
- **CSS:** Custom styling for the application interface.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python packages: `streamlit`, `Pillow`, `tensorflow`, `numpy`

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/9TaiLBeasT/CancerForecast.git
    cd CancerForecast
    ```

2. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Download the pre-trained models:**

    Ensure that you have the model files in the respective directory:
    
    - Models for Brain Cancer, Acute Lymphoblastic Leukemia, Breast Cancer, Cervical Cancer, Kidney Cancer, Lung and Colon Cancer

    Place these model files in the appropriate directories or update the file paths in the code.

### Usage

1. **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

2. **Open the application:**

    Navigate to the local server URL provided by Streamlit, typically `http://localhost:8501`.

3. **Upload Images:**

    Use the file uploader to upload medical images. The system will display predictions and explanations based on the uploaded images.

## Code Structure

- `app.py`: Main application script containing the Streamlit app logic.
- `styles/main.css`: Custom CSS for styling the application interface.
- `./(cancer name)/`: Directory containing model files and example images.

## Adding New Models

To add new models (e.g., for Brain Cancer, Acute Lymphoblastic Leukemia, etc.):

1. **Update the code to include the new models:**

    Ensure that you have the model files and update the file paths in the code accordingly.

2. **Update the `CLASS_NAMES` and `EXPLANATIONS` dictionaries:**

    Add the new classes and their corresponding explanations.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have suggestions or improvements.


## Video Presentation

[![Video Presentation](https://img.youtube.com/vi/yourvideoid/maxresdefault.jpg)](https://drive.google.com/file/d/1vsZpdsKgJxZ2oq9c-OBxGhiDJHmpN523/view?usp=sharing)

---

For any questions or further information, feel free to contact me via GitHub or email.

