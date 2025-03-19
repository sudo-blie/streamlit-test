# Description: Simple Streamlit app for capturing an image from webcam or uploading an image file
import streamlit as st
from PIL import Image

def main():
    st.title("Image Capture Demo")
    
    # Option to capture image from webcam
    st.subheader("Capture Image from Webcam")
    webcam_image = st.camera_input("Take a picture")

    # Option to upload an image file
    st.subheader("Or Upload an Image")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Determine which image source to use and display it
    image_source = None
    if webcam_image:
        image_source = webcam_image
        st.success("Image captured from webcam!")
    elif uploaded_image:
        image_source = uploaded_image
        st.success("Image uploaded successfully!")

    if image_source:
        # Display the image with caption
        st.subheader("Captured/Uploaded Image")
        st.image(image_source, caption='Selected Image', use_container_width=True)

        # Display some basic image information
        try:
            img = Image.open(image_source)
            st.text(f"Image Size: {img.size}")
            st.text(f"Image Mode: {img.mode}")
        except Exception as e:
            st.error(f"Error reading image details: {str(e)}")

if __name__ == "__main__":
    main()