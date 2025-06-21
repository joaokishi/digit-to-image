import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os

# --- 1. Define Model Architecture (must be identical to the training script) ---
# It's crucial that this class definition matches the one used for training.
LATENT_DIM = 20
NUM_CLASSES = 10

class CVAE(nn.Module):
    def __init__(self, feature_dim, latent_dim, num_classes):
        super(CVAE, self).__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.encoder_fc1 = nn.Linear(feature_dim + num_classes, 512)
        self.encoder_fc2 = nn.Linear(512, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_log_var = nn.Linear(256, latent_dim)
        self.decoder_fc1 = nn.Linear(latent_dim + num_classes, 256)
        self.decoder_fc2 = nn.Linear(256, 512)
        self.decoder_fc3 = nn.Linear(512, feature_dim)

    def encode(self, x, y):
        inputs = torch.cat([x, y], 1)
        h = F.relu(self.encoder_fc1(inputs))
        h = F.relu(self.encoder_fc2(h))
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        inputs = torch.cat([z, y], 1)
        h = F.relu(self.decoder_fc1(inputs))
        h = F.relu(self.decoder_fc2(h))
        return torch.sigmoid(self.decoder_fc3(h))

    def forward(self, x, y):
        mu, log_var = self.encode(x.view(-1, self.feature_dim), y)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z, y)
        return reconstruction, mu, log_var

# --- 2. Load the Trained Model ---
# Use a cache to load the model only once.
@st.cache_resource
def load_model():
    model_path = 'cvae_mnist.pth'
    if not os.path.exists(model_path):
        st.error(f"Model file not found! Please place '{model_path}' in the same directory as this script.")
        return None
    
    device = torch.device("cpu") # Run inference on CPU
    model = CVAE(feature_dim=28*28, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set model to evaluation mode
    return model

model = load_model()

# --- 3. Web Application UI with Streamlit ---
st.set_page_config(layout="wide")
st.title("Handwritten Digit Generation using a CVAE")
st.write("This app uses a Conditional Variational Autoencoder (CVAE) trained on the MNIST dataset to generate new images of handwritten digits.")

st.sidebar.header("Controls")
selected_digit = st.sidebar.selectbox("Select a digit to generate (0-9):", list(range(10)))

if model:
    if st.sidebar.button("Generate 5 Images"):
        st.subheader(f"Generating 5 images for the digit: {selected_digit}")
        
        # Create 5 columns to display images side-by-side
        cols = st.columns(5)
        
        for i in range(5):
            with torch.no_grad():
                # 1. Sample a random vector from the latent space (standard normal distribution)
                z = torch.randn(1, LATENT_DIM)
                
                # 2. Create the condition label (the digit we want) as a one-hot vector
                label = torch.tensor([selected_digit])
                label_one_hot = F.one_hot(label, num_classes=NUM_CLASSES).float()
                
                # 3. Generate the image using the decoder part of the CVAE
                generated_img_tensor = model.decode(z, label_one_hot)
                
                # 4. Reshape and convert the output tensor to a displayable image
                img_array = generated_img_tensor.view(28, 28).cpu().numpy()
                
                # Display the image in the corresponding column
                with cols[i]:
                    st.image(img_array, caption=f"Generated #{i+1}", width=150)

    st.sidebar.info("Click the button above to generate new digit images. Each click will produce a new set of 5 unique images.")

else:
    st.warning("Model could not be loaded. Please check the error message above.")
