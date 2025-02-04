import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Constants
TARGET_SIZE = (64, 64)  # Model's expected input size

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

model = load_inpainting_model()

def preprocess_image(image):
    """Convert image to 64x64, normalize, and return as numpy array."""
    image = image.resize(TARGET_SIZE, Image.LANCZOS)  # High-quality resize
    # image = np.array(image).astype(np.float32) / 255.0
    return image

def upscale_image(image, size=(200, 200)):
    """Upscale 64x64 image to 200x200 using high-quality interpolation."""
    image = (image * 255).astype(np.uint8)  # Convert back to 0-255 range
    image = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)  # High-quality upscaling
    return image

def create_square_mask(image, x, y, patch_size=8):
    """Create a square mask at given (x,y) position."""
    mask = np.ones_like(image)
    mask[y:y+patch_size, x:x+patch_size, :] = 0
    return mask

# Sidebar Instructions
st.sidebar.title("📖 How to Use")
st.sidebar.markdown("""
1️⃣ **Upload an Image** (JPG, PNG, JPEG)  
2️⃣ **Draw a Mask** on the image where you want inpainting.  
3️⃣ **Click Process** to restore the missing area.  
4️⃣ **Compare Output** between the original and restored images.  
""")
st.sidebar.markdown("---")
st.sidebar.subheader("✨ Features")
st.sidebar.markdown("""
✅ Supports **high-quality inpainting**  
✅ Processes **64x64 images** and upscales to **200x200**  
✅ Uses **deep learning** for accurate restoration  
""")

# Streamlit UI
st.title("🎨 Image Inpainting Demo")
st.markdown("**AI-powered tool to restore missing parts of an image.**")

uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    try:
        # Load and preprocess image
        image = Image.open(uploaded_file).convert('RGB')
        processed_image = preprocess_image(image)

        # Create three columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📥 Input Image (64x64)")
            st.image(processed_image, caption="Uploaded Image", use_column_width=True)

            # Canvas for user to draw mask
            st.subheader("✏️ Draw on the Image")
            st.markdown("Click to draw the missing area for inpainting.")
            if processed_image is not None:
                canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 1)",
                stroke_width=0,
                stroke_color="rgba(255, 255, 255, 1)",
                background_image=processed_image,
                update_streamlit=True,
                drawing_mode="freedraw",
                key="canvas"
                )
else:
    st.error("❌ Error: No image loaded.")

        with col2:
            if canvas_result.json_data is not None and canvas_result.json_data["objects"]:
                last_point = canvas_result.json_data["objects"][-1]
                
                # Scale coordinates
                x = int(last_point["left"])
                y = int(last_point["top"])

                if model is not None:
                    mask = create_square_mask(processed_image, x, y)
                    masked_image = processed_image * mask
                    model_input = np.expand_dims(masked_image, axis=0)
                    restored = model.predict(model_input)[0]

                    # Upscale images for display
                    masked_image_upscaled = upscale_image(masked_image)
                    restored_upscaled = upscale_image(restored)

                    st.subheader("🔍 Masked Image (200x200)")
                    st.image(masked_image_upscaled, caption="Masked Image", use_column_width=True)

                    with col3:
                        st.subheader("✅ Restored Image (200x200)")
                        st.image(restored_upscaled, caption="Restored Image", use_column_width=True)
                else:
                    st.warning("⚠️ Model not loaded. Please check if the model file exists.")

    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")

st.markdown("📢 **Note:** This is a demo application. Image quality depends on the model's training.")
