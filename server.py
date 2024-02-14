import streamlit as st
import numpy as np
import tensorflow as tf
from fpdf import FPDF  # Import FPDF library for PDF generation
import datetime

# Check TensorFlow version
tf_version = tf.__version__
st.write(f"Using TensorFlow version: {tf_version}")

# Load models without compiling
fake_model = tf.keras.models.load_model('https://drive.google.com/file/d/1VedDwkNP8FLukDmJOvfzh517bZPSNo2j/view?usp=drive_link', compile=False)
real_model = tf.keras.models.load_model('https://drive.google.com/open?id=10suO7bxu62nswnZRAwz95AobOUQq49Cx&usp=drive_copy', compile=False)

# Recompile with a compatible optimizer
fake_model.compile(optimizer='adam')  # Replace with a compatible optimizer if needed
real_model.compile(optimizer='adam')  # Replace with a compatible optimizer if needed

# Create a wrapper model to handle different input shapes
class WrapperModel:
    def __init__(self, real_model, fake_model):
        self.real_model = real_model
        self.fake_model = fake_model

    def predict(self, image):
        resized_image = image.resize((224, 224)) if self.real_model is not None else image.resize((256, 256))
        array = tf.keras.preprocessing.image.img_to_array(resized_image)
        array = np.expand_dims(array, axis=0)
        if self.real_model is not None:
            return self.real_model.predict(array)
        else:
            return self.fake_model.predict(array)

# Create the wrapper model
wrapper_model = WrapperModel(real_model, fake_model)

st.title("Deepfake Image Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = tf.keras.preprocessing.image.load_img(uploaded_file)  # Load image without resizing
    prediction = wrapper_model.predict(image)  # Let the wrapper handle resizing and prediction

    # Combine predictions (adjust logic if needed)
    final_prediction = "Fake" if prediction > 0.5 else "Real"  # Example threshold for single output value

    st.image(uploaded_file, caption=f"Prediction: {final_prediction}")

    user_name = st.text_input("Enter your name:")  # Get user input for report details

    # Get current date and time
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Choose your desired font and apply it to the PDF content
    pdf_font = 'Times'  # Change this to your preferred font (e.g., 'Helvetica', 'Arial')

    # Create report content in a list for better formatting
    report_content = [
        f"           Fake Image Detection Report",
        "",
        f"User Name: {user_name}",
        f"Media Type: Image",
        f"Prediction: {final_prediction}",
        f"Prediction Accuracy: {prediction[0][0]:.4f}",
        f"Date and Time: {current_datetime}",
        f"Image File Name: {uploaded_file.name}",
    ]

    # Generate PDF report with improved formatting and chosen font
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font(pdf_font, 'B', 16)

    # Iterate through report_content and add each line with formatting
    for line in report_content:
        if line == report_content[0]:  # Bold title
            pdf.cell(40, 10, line, align='C', ln=1)
        elif line == "":  # Add vertical space
            pdf.cell(40, 10, "", ln=1)
        else:
            pdf.cell(40, 10, line, ln=1)

    pdf_bytes = pdf.output(dest="S").encode("latin-1")

    # Add button to download report
    if st.button("Generate Report"):
        st.download_button(
            label="Download Report",
            data=pdf_bytes,
            file_name="report.pdf"
        )

    # Store current date and time
