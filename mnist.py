import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# --- 1. Define Hyperparameters and Device Configuration ---
# Training settings that you can tune
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
LATENT_DIM = 20 # The dimensionality of the latent space
NUM_CLASSES = 10 # Digits 0-9

# Set the device to GPU (T4) if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 2. Model Architecture: Conditional Variational Autoencoder (CVAE) ---
class CVAE(nn.Module):
    """
    CVAE architecture. The model is conditioned on the digit label,
    allowing us to control which digit to generate.
    """
    def __init__(self, feature_dim, latent_dim, num_classes):
        super(CVAE, self).__init__()
        
        self.feature_dim = feature_dim # 28*28 = 784
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # --- Encoder ---
        # It takes the image and its one-hot encoded label as input
        self.encoder_fc1 = nn.Linear(feature_dim + num_classes, 512)
        self.encoder_fc2 = nn.Linear(512, 256)
        # These two layers output the parameters of the latent distribution
        self.fc_mu = nn.Linear(256, latent_dim) # for the mean (mu)
        self.fc_log_var = nn.Linear(256, latent_dim) # for the log variance (log_var)

        # --- Decoder ---
        # It takes a sample from the latent space and a one-hot label as input
        self.decoder_fc1 = nn.Linear(latent_dim + num_classes, 256)
        self.decoder_fc2 = nn.Linear(256, 512)
        self.decoder_fc3 = nn.Linear(512, feature_dim)

    def encode(self, x, y):
        """Encodes the input image and label into latent space parameters."""
        # Concatenate the flattened image and the one-hot encoded label
        inputs = torch.cat([x, y], 1)
        h = F.relu(self.encoder_fc1(inputs))
        h = F.relu(self.encoder_fc2(h))
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        """
        The reparameterization trick: allows gradients to flow through the
        stochastic sampling process.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) # sample from a standard normal distribution
        return mu + eps * std

    def decode(self, z, y):
        """Decodes a latent vector and a label back into an image."""
        # Concatenate the latent vector and the one-hot encoded label
        inputs = torch.cat([z, y], 1)
        h = F.relu(self.decoder_fc1(inputs))
        h = F.relu(self.decoder_fc2(h))
        # Use sigmoid to ensure output pixel values are between 0 and 1
        return torch.sigmoid(self.decoder_fc3(h))

    def forward(self, x, y):
        """The full forward pass of the CVAE."""
        mu, log_var = self.encode(x.view(-1, self.feature_dim), y)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z, y)
        return reconstruction, mu, log_var


# --- 3. Loss Function ---
def loss_function(recon_x, x, mu, log_var):
    """
    Calculates the VAE loss, which is a sum of two parts:
    1. Reconstruction Loss: How well the decoded image matches the original.
    2. KL Divergence: A regularizer that forces the latent space to be smooth.
    """
    # Use Binary Cross-Entropy for reconstruction loss
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # KL divergence formula: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return BCE + KLD


# --- 4. Data Loading and Training Loop ---
def train():
    # Load MNIST Dataset
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model and optimizer
    model = CVAE(feature_dim=28*28, latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            # Convert labels to one-hot encoding
            labels_one_hot = F.one_hot(labels, num_classes=NUM_CLASSES).float().to(device)

            optimizer.zero_grad()
            
            recon_batch, mu, log_var = model(data, labels_one_hot)
            loss = loss_function(recon_batch, data, mu, log_var)
            
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        avg_loss = train_loss / len(train_loader.dataset)
        print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')

    # --- 5. Save the Trained Model ---
    model_path = 'cvae_mnist.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    print("You can now download this file and use it with the Streamlit app.")

if __name__ == '__main__':
    train()
