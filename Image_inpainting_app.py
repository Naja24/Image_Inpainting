import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# Load the trained model
@st.cache_resource
def load_inpainting_model():
    model_path = "New_Model1.h5"
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please ensure the model file exists.")
        return None
    try:
        model = load_model(model_path, compile=False)
        model.compile(optimizer="adam", loss="mse")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None


# Load model
model = load_inpainting_model()


def preprocess_image(image, target_size=(64, 64)):
    """
    Preprocess image to match model's expected input
    """
    if isinstance(image, Image.Image):
        image = image.convert('RGB')
        image = np.array(image)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    return image


def create_square_mask(image, x, y, patch_size=(8, 8)):
    """Create a square mask at the specified position."""
    mask = np.ones_like(image)
    mask[y:y + patch_size[1], x:x + patch_size[0], :] = 0
    return mask


# Streamlit UI
st.title("Image Inpainting Demo 🎨")
st.markdown("""
    **Welcome to the Image Inpainting Tool!**  
    This tool allows you to upload an image, mask a 8x8 portion of it, and see how the model inpaints the missing parts.
""")

# Sidebar for instructions and preprocessing details
with st.sidebar:
    st.header("Instructions 📝")
    st.markdown("""
        1. **Upload an image** using the file uploader below.
        2. **Click on the left image** to place a square mask for inpainting.
        3. The **masked image** and **restored image** will appear on the right.
    """)
    st.divider()
    st.header("Preprocessing Steps")
    st.markdown("""
        Before the image is fed into the model, it undergoes the following steps:
        - **Resizing**: The image is resized to 64x64 pixels.
        - **Normalization**: Pixel values are scaled to the range [0, 1].
        - **Masking**: A square mask is applied to the selected region.
    """)

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    try:
        # Read and process image
        image = Image.open(uploaded_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            image = image.resize((64, 64))
        processed_image = preprocess_image(image)

        # Create three columns for input, masked, and restored images
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Input Image")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Input image with drawing canvas
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 1)",
                stroke_width=0,
                stroke_color="rgba(255, 255, 255, 1)",
                background_image=image,
                drawing_mode="point",
                key="canvas",
                width=200,
                height=200
            )

        with col2:
            if canvas_result.json_data is not None and canvas_result.json_data["objects"]:
                # Get clicked point
                last_point = canvas_result.json_data["objects"][-1]

                # Scale coordinates
                scale_factor = 64 / 200
                x = int(last_point["left"] * scale_factor)
                y = int(last_point["top"] * scale_factor)

                # Process and restore image
                if model is not None:
                    mask = create_square_mask(processed_image, x, y)
                    masked_image = processed_image * mask
                    model_input = np.expand_dims(masked_image, axis=0)
                    restored = model.predict(model_input)[0]

                    # Show masked image
                    st.subheader("Masked Image")
                    st.image(cv2.resize(masked_image, (200, 200)),
                             caption="Masked Image",
                             use_container_width=False)

                    with col3:
                        # Show restored image
                        st.subheader("Restored Image")
                        st.image(cv2.resize(restored, (200, 200)),
                                 caption="Restored Image",
                                 use_container_width=False)

                else:
                    st.warning("Model not loaded. Please check if the model file exists.")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# Footer
st.divider()
st.markdown("""
    **Note**: This is a demo application. The model's performance depends on the quality of the trained model.
""")