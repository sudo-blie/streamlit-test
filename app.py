# Description: Streamlit app for capturing an image from webcam or uploading an image file, and performing OCR using Gemini API.
import streamlit as st
from PIL import Image
import tempfile
import os
from dotenv import load_dotenv
from gemma_ocr import perform_ocr

# Load environment variables
load_dotenv()


def save_bytes_to_temp_file(image_bytes):
    """Save image bytes to a temporary file and return the path."""
    # Create a temporary file with .jpg extension
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    try:
        temp_file.write(image_bytes)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        temp_file.close()
        os.unlink(temp_file.name)
        raise e

def process_image(image_source):
    """Process image source and perform OCR."""
    try:
        # Create temporary file from image bytes
        temp_file_path = save_bytes_to_temp_file(image_source.getvalue())
        
        try:
            # Perform OCR using gemma_ocr
            result = perform_ocr(temp_file_path)
            return result
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def main():
    st.title("Image Capture and OCR with Google GEMMA")
    
    # Check if Ollama host is configured
    if not os.getenv("OLLAMA_HOST"):
        st.warning("Please set the OLLAMA_HOST environment variable")
        return

    # Option to capture image from webcam
    st.subheader("Capture Image from Webcam")
    webcam_image = st.camera_input("Take a picture")

    # Option to upload an image file
    st.subheader("Or Upload an Image")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Determine which image source to use
    image_source = None
    if webcam_image:
        image_source = webcam_image
    elif uploaded_image:
        image_source = uploaded_image

    if image_source:
        # Display the image
        st.image(image_source, caption='Selected Image', use_container_width=True)

        # Perform OCR
        with st.spinner('Performing OCR...'):
            ocr_result = process_image(image_source)
            if ocr_result:
                st.success('OCR Completed!')
                
                # Display OCR results
                st.subheader("Raw OCR Extracted Text")
                # Display each key-value pair from the OCR result
                for key, value in ocr_result.items():
                    st.text_input(f"{key}", value, disabled=True)

if __name__ == "__main__":
    main()
