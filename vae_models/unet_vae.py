import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import os



class UNetVAE(nn.Module):
    MODEL_DIR = "models"
    def __init__(self, latent_dim=512):
        super(UNetVAE, self).__init__()

        # Encoder - Use ResNet34 as backbone
        # resnet = models.resnet34(pretrained=False)
        resnet = models.resnet34(weights=None)
        # Modify the first convolutional layer to accept 1 channel (grayscale)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layers

        # Bottleneck - Latent space
        self.fc_mu = nn.Linear(512 * (512 // 32) * (1024 // 32), latent_dim)  # ResNet reduces size by factor of 32
        self.fc_logvar = nn.Linear(512 * (512 // 32) * (1024 // 32), latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 512 * (512 // 32) * (1024 // 32))

        # Decoder - Transposed convolutions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16x32 -> 32x64
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x64 -> 64x128
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x128 -> 128x256
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128x256 -> 256x512
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 256x512 -> 512x1024
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),  # No stride, keeps size
            nn.Sigmoid()  # Ensures output is in range [0,1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        enc = self.encoder(x)
        enc = enc.view(enc.size(0), -1)

        mu, logvar = self.fc_mu(enc), self.fc_logvar(enc)
        z = self.reparameterize(mu, logvar)

        dec_input = self.fc_decode(z).view(-1, 512, 16, 32)  # Reshape to match last encoder layer
        recon = self.decoder(dec_input)

        print(f"Input x shape: {x.shape}")  # Debugging
        print(f"Reconstructed x shape: {recon.shape}")  # Debugging

        return recon, mu, logvar

    @staticmethod
    # ELBO Loss function (outside the UNetVAE class)
    def vae_loss(recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div

    @staticmethod
    def train_model(dataloader, model_name = "unet_vae_ultrasound.pth", num_epochs=50):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        vae = UNetVAE().to(device)
        optimizer = optim.Adam(vae.parameters(), lr=1e-4)

        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0
            for images in dataloader:
                images = images.to(device)
                optimizer.zero_grad()

                recon_images, mu, logvar = vae(images)
                loss = UNetVAE.vae_loss(recon_images, images, mu, logvar)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader.dataset):.4f}")

        # Save model
        # Ensure the 'models' directory exists (will create it if it doesn't exist)
        os.makedirs(UNetVAE.MODEL_DIR, exist_ok=True)
        torch.save(vae.state_dict(), os.path.join(UNetVAE.MODEL_DIR, model_name))

    @staticmethod
    def load_model(latent_dim=512, model_name="unet_vae_ultrasound.pth"):
        # Initialize model
        vae = UNetVAE(latent_dim=latent_dim)
        path = os.path.join(UNetVAE.MODEL_DIR, model_name)

        # Check if the weights file exists
        if os.path.exists(path):
            vae.load_state_dict(torch.load(path, weights_only=True))
            vae.eval()  # Set the model to evaluation mode
            print(f"Model loaded successfully from {path}")
        else:
            print(f"Error: The weights file {path} does not exist.")

        return vae