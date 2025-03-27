# Image_Inpainting
This project leverages Generative Adversarial Networks (GANs) and Convolutional Autoencoders to restore missing regions in images by predicting pixel values.

🔗 Live Demo: https://imageinpainting-autoencoder.streamlit.app/

✨ Features
GAN-powered Inpainting: Improved image restoration using a Generator-Discriminator framework.

Convolutional Autoencoder: Learns image representations for pixel-wise reconstruction.

Enhanced Performance: Trained with adversarial and perceptual losses for realistic inpainting.

Streamlit Web App: Easily upload and process images online.

📌 Model Overview
1️⃣ Generator (Image Restoration)
Uses a U-Net like architecture to reconstruct missing patches.

Learns spatial correlations in images for realistic outputs.

2️⃣ Discriminator (Adversarial Training)
Classifies images as real or generated to improve quality.

Helps refine textures and details in reconstructed regions.

🚀 How to Use
Upload an image with missing parts.

Model predicts missing pixel values.

Download the restored image.

🛠️ Installation
Clone the repository and install dependencies:

bash
Copy
Edit
git clone https://github.com/Naja24/Image_Inpainting.git  
cd Image_Inpainting  
pip install -r requirements.txt  
Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py  
📜 Training Details
Dataset: Trained on diverse datasets with missing patches.

Training Strategy:

Generator minimizes perceptual and adversarial losses.

Discriminator ensures high-quality restoration.

Uses TensorFlow/Keras for implementation.

📊 Results
GAN-based training leads to sharper, more natural inpainting.

Perceptual loss helps in generating visually appealing textures.

🤝 Contributing
Feel free to fork and improve the model! PRs are welcome.
