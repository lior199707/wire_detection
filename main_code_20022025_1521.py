import os
import csv
import json
import cv2  # noqa: F401
import torch
from dataset_loader.ultrasound_dataset import UltrasoundDataset
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # For CSV output

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

import random
from datetime import datetime  # For timestamped folder

# New
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from piq import FID
import pywt

###############################################################################
# Global Parameters
###############################################################################

# TODO: ADD EVALUATION MATRICES DIR
TIMESTAMP = datetime.now().strftime("%d%m%Y_%H%M")
RESULTS_FOLDER_NAME = f"vae_experiment_{TIMESTAMP}"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
MODEL_DIR = "models"
RESULTS_DIR = "results"
ANOMALY_RESULTS_DIR = "anomaly_results"

DEFAULT_RESIZE_HEIGHT = 512
DEFAULT_RESIZE_WIDTH = 1024
DEFAULT_LATENT_DIM = 256
DEFAULT_NUM_EPOCHS = 1000
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_STEP_SIZE = 100
DEFAULT_GAMMA = 0.5
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_TRAINING_IMAGES = None
DEFAULT_NUM_SAMPLES_TO_SHOW = 4
# Best model name
DEFAULT_MODEL_NAME = "conv_vae.pt"
CONFIG_FILE_NAME = "dataset_config.json"
ROOT_DATA_DIR = "data"
LABEL_TRAIN = "no_wire"
LABEL_ANOMALY = "wire"
EARLY_STOPPING_PATIENCE = 20
ANOMALY_VISUALIZATION_SAMPLES_PER_EPOCH = 10

###############################################################################
# Misc Setup
###############################################################################

torch.backends.cudnn.benchmark = True
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The parameter 'pretrained' is deprecated since 0.13"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Arguments other than a weight enum or `None` for 'weights'"
)

###############################################################################
# Dataset creation
###############################################################################

def create_datasets():
    """
    Creates the training (no_wire) and anomaly (wire) datasets using the UltrasoundDataset class.
    """
    custom_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((DEFAULT_RESIZE_HEIGHT, DEFAULT_RESIZE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    full_dataset = UltrasoundDataset(
        root_dir=ROOT_DATA_DIR,
        config_file=CONFIG_FILE_NAME,
        label=LABEL_TRAIN,
        transform=custom_transform
    )
    print("Full dataset size (no_wire):", len(full_dataset))

    anomaly_dataset = UltrasoundDataset(
        root_dir=ROOT_DATA_DIR,
        config_file=CONFIG_FILE_NAME,
        label=LABEL_ANOMALY,
        transform=custom_transform
    )
    print("Anomaly dataset size (wire):", len(anomaly_dataset))

    return full_dataset, anomaly_dataset

###############################################################################
# Reconstruction Evaluator
###############################################################################
# NEW
class VAEEvaluator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.lpips_loss = lpips.LPIPS(net="vgg").to(device)  # Perceptual loss
        self.fid_metric = FID().to(device)  # FID Score
    
    def evaluate(self, batch_data, recon, mu, logvar):
        batch_size = batch_data.size(0)
        total_mse = 0
        total_psnr = 0
        total_ssim = 0
        total_lpips = 0
        total_ifc = 0
        #////////////////
        kl_divergence = self.kl_divergence(mu,logvar)
        mutual_information = self.mutual_information(mu, logvar)
        total_correlation = self.total_correlation(mu, logvar)

        for i in range(batch_size):
                    # if samples_shown >= num_samples and num_samples != -1:
                    #     break

                    orig_img = batch_data[i].cpu().numpy().squeeze()
                    recon_img = recon[i].cpu().numpy().squeeze()
                    total_mse += self.mse(orig_img, recon_img)
                    total_psnr += self.psnr(orig_img, recon_img)
                    total_ssim += self.ssim(orig_img, recon_img)
                    total_lpips += self.lpips(orig_img, recon_img)
                    total_ifc += self.ifc(orig_img, recon_img)
        # return total_mse/batch_size, total_psnr/batch_size, total_ssim/batch_size, total_lpips/batch_size, total_ifc/batch_size, kl_divergence, mutual_information, total_correlation
        return total_mse, total_psnr, total_ssim, total_lpips, total_ifc, kl_divergence, mutual_information, total_correlation

    @staticmethod
    def mse(original, reconstructed):
        return np.mean((original - reconstructed) ** 2)

    @staticmethod
    def psnr(original, reconstructed):
        return psnr(original, reconstructed, data_range=1.0)
                

    @staticmethod
    def ssim(original, reconstructed):
        return ssim(original, reconstructed, data_range=1.0)

    def lpips(self, original, reconstructed):
        original_torch = torch.tensor(original).unsqueeze(0).unsqueeze(0).to(self.device)
        reconstructed_torch = torch.tensor(reconstructed).unsqueeze(0).unsqueeze(0).to(self.device)
        return self.lpips_loss(original_torch, reconstructed_torch).item()


    def fid(self, original_batch, reconstructed_batch):
        original_torch = torch.tensor(original_batch).unsqueeze(1).to(self.device)
        reconstructed_torch = torch.tensor(reconstructed_batch).unsqueeze(1).to(self.device)
        return self.fid_metric(original_torch, reconstructed_torch).item()
    
    @staticmethod
    def ifc(original, reconstructed):
        """
        Compute Information Fidelity Criterion (IFC) based on wavelet coefficients.
        Normalizes wavelet coefficients before calculation.
        """
        def compute_wavelet_coeffs(image):
            coeffs = pywt.wavedec2(image, 'haar', level=4)
            return coeffs[0]  # Approximate coefficients

        # Compute wavelet coefficients
        orig_coeffs = compute_wavelet_coeffs(original)
        recon_coeffs = compute_wavelet_coeffs(reconstructed)

        # Normalize the coefficients (zero mean, unit variance)
        def normalize(coeffs):
            coeffs_mean = np.mean(coeffs)
            coeffs_std = np.std(coeffs)
            return (coeffs - coeffs_mean) / (coeffs_std + 1e-8)

        orig_coeffs = normalize(orig_coeffs)
        recon_coeffs = normalize(recon_coeffs)

        # Compute means and variances
        mu_x, sigma_x = np.mean(orig_coeffs), np.var(orig_coeffs)
        mu_y, sigma_y = np.mean(recon_coeffs), np.var(recon_coeffs)

        # Mutual information estimation
        cov_xy = np.cov(orig_coeffs.flatten(), recon_coeffs.flatten())[0, 1]
        ifc_value = (2 * mu_x * mu_y + 1e-8) * (2 * cov_xy + 1e-8) / ((mu_x**2 + mu_y**2 + 1e-8) * (sigma_x + sigma_y + 1e-8))

        return ifc_value
    

    @staticmethod
    def kl_divergence(mu, logvar):
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    # A more robust approach is to use log(1 + |latent|), which avoids shifting and is used in some mutual information estimators
    # @staticmethod
    # def mutual_information(latent):
    #     latent = torch.log(1 + torch.abs(latent))  # Safe log
    #     res = torch.mean(latent)
    #     return res
    
    @staticmethod
    def mutual_information(mu, logvar):
        """
        Compute the Mutual Information (MI) between latent variables and data
        for a VAE with a Gaussian latent space.
       
        This approximation assumes the posterior q(z|x) ~ N(mu, sigma^2)
        and the prior p(z) ~ N(0, I).
        """
        # Entropy of the Gaussian distribution q(z|x) (approximation)
        log_two_pi = torch.log(torch.tensor(2 * torch.pi))  # Make sure this is a tensor
        entropy = 0.5 * torch.sum(1 + log_two_pi + logvar, dim=1)

        # KL Divergence between q(z|x) and p(z) = N(0, I)
        kl_divergence = 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1. - logvar, dim=1)

        # Mutual Information: MI = H(q(z|x)) - KL(q(z|x) || p(z))
        mi = entropy - kl_divergence

        return torch.mean(mi)
    
    # @staticmethod
    # def total_correlation(latent):
    #     # Compute log densities
    #     log_qz = torch.logsumexp(latent, dim=0) - np.log(latent.shape[0])
    #     log_qzi = torch.logsumexp(latent, dim=1).mean(dim=0) - np.log(latent.shape[1])
    
    #     # TC is the KL divergence between joint and independent marginal distributions
    #     return (log_qz - log_qzi).sum()

    @staticmethod 
    def total_correlation(mu, logvar): 
        batch_size, latent_dim = mu.shape 
        log_qz = torch.logsumexp(-0.5 * ((mu ** 2) + logvar.exp()), dim=0) - np.log(batch_size) 
        log_qzi = torch.mean(torch.logsumexp(-0.5 * ((mu ** 2) + logvar.exp()), dim=1)) - np.log(latent_dim) 
        return torch.sum(log_qz - log_qzi)  

###############################################################################
# Model Definition: Convolutional Variational Autoencoder
###############################################################################

class ConvVAE(nn.Module):
    MODEL_DIR = MODEL_DIR

    def __init__(self, input_height=DEFAULT_RESIZE_HEIGHT, input_width=DEFAULT_RESIZE_WIDTH, latent_dim=DEFAULT_LATENT_DIM):
        super().__init__()
        self.input_height = input_height
        self.input_width = input_width
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Determine flattened size after convolutions
        test_input = torch.randn(1, 1, input_height, input_width)
        conv_out_shape = self.encoder_conv(test_input).shape
        flattened_size = np.prod(conv_out_shape[1:])

        self.encoder_fc = nn.Linear(flattened_size, latent_dim * 2)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, flattened_size)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
        self.final_resize = nn.AdaptiveAvgPool2d((input_height, input_width))

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        mu, logvar = torch.chunk(self.encoder_fc(x), 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_fc(z)
        # Re-infer shape from test pass
        test_input = torch.randn(1, 1, self.input_height, self.input_width).to(z.device)
        conv_out_shape = self.encoder_conv(test_input).shape
        x = x.view(x.size(0), conv_out_shape[1], conv_out_shape[2], conv_out_shape[3])
        x = self.decoder_conv(x)
        return self.final_resize(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @staticmethod
    def loss_function(x, recon_x, mu, logvar):
        """
        Standard VAE loss = MSE reconstruction loss + KL divergence.
        """
        batch_size = x.size(0)
        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / batch_size
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar)) / batch_size
        return recon_loss + kl_loss

    @staticmethod
    def train_model(
        train_loader, 
        val_loader=None, 
        anomaly_loader=None,
        input_height=DEFAULT_RESIZE_HEIGHT, 
        input_width=DEFAULT_RESIZE_WIDTH, 
        latent_dim=DEFAULT_LATENT_DIM,
        num_epochs=DEFAULT_NUM_EPOCHS, 
        learning_rate=DEFAULT_LEARNING_RATE,
        step_size=DEFAULT_STEP_SIZE, 
        gamma=DEFAULT_GAMMA, 
        model_name=DEFAULT_MODEL_NAME,
        device=None, 
        num_samples_to_show=DEFAULT_NUM_SAMPLES_TO_SHOW,
        results_folder=None, 
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        anomaly_samples_per_epoch=ANOMALY_VISUALIZATION_SAMPLES_PER_EPOCH
    ):
        """
        Trains the VAE model using the specified data loaders and parameters.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = ConvVAE(input_height=input_height, input_width=input_width, latent_dim=latent_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        if results_folder is None:
            timestamp = datetime.now().strftime("%d%m%Y_%H%M")
            results_folder = os.path.join(RESULTS_DIR, f"vae_results_{timestamp}")
        os.makedirs(results_folder, exist_ok=True)

        dataset_size = len(train_loader.dataset)
        best_val_loss = float('inf')
        best_model_wts = None
        epochs_no_improve = 0
        training_log = []

        # Save hyperparameters
        hyperparams = {
            "input_height": input_height,
            "input_width": input_width,
            "latent_dim": latent_dim,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "step_size": step_size,
            "gamma": gamma,
            "batch_size": train_loader.batch_size,
            "max_training_images": DEFAULT_MAX_TRAINING_IMAGES,
            "model_name": model_name,
            "early_stopping_patience": early_stopping_patience,
            "anomaly_samples_per_epoch": anomaly_samples_per_epoch
        }
        with open(os.path.join(results_folder, "hyperparameters.json"), "w") as f:
            json.dump(hyperparams, f, indent=4)

        print("--------------------------------------------------------------------------------------")
        print(f"Results will be saved to: {results_folder}")

        # TODO: CHANGE THIS FOR SAVING THE MODEL OF EACH IMPROVED ITERATION
        os.makedirs(ConvVAE.MODEL_DIR, exist_ok=True)
        models_of_experiment_dir = os.path.join(MODEL_DIR, RESULTS_FOLDER_NAME) # models/vae_exp_342342
        os.makedirs(models_of_experiment_dir, exist_ok=True)
        # os.makedirs(ConvVAE.MODEL_DIR, exist_ok=True)
        # TODO: models_of_experiment_dir = os.path.join(MODEL_DIR, RESULTS_FOLDER_NAME) # models/vae_exp_342342
        best_model_dir = os.path.join(models_of_experiment_dir, "best_model")
        os.makedirs(best_model_dir, exist_ok=True)

        metrics_file = os.path.join(results_folder, "evaluation_metrics.csv")
        print(f"Metrics will be saved to: {metrics_file}")

        print(f"models will be saved to: {models_of_experiment_dir}")
        print("--------------------------------------------------------------------------------------")
        

        # ---------------- Training Loop ---------------- #
        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            epoch_loss = 0.0
            model.train()

            for batch_data in tqdm(train_loader, desc="Training"):
                batch_data = batch_data.to(device)
                optimizer.zero_grad()
                recon, mu, logvar = model(batch_data)
                loss = ConvVAE.loss_function(batch_data, recon, mu, logvar)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            avg_train_loss = epoch_loss / dataset_size if dataset_size else 0
            current_lr = scheduler.get_last_lr()[0]
            print("//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
            print(f"  > Epoch {epoch+1} Train Avg Loss: {avg_train_loss:.4f}, LR: {current_lr:.6f}")

            # Track results
            epoch_results = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': None,
                'lr': current_lr
            }

            # Validation
            if val_loader is not None:
                val_loss, mse, psnr, ssim, lpips, ifc, kl, mi, tc = ConvVAE.validate_model(model, val_loader, device)
                
                print(f"  > Epoch {epoch+1}\n    Validation Avg Loss: {val_loss:.4f}")
                print(f"    MSE={mse:.4f}, PSNR={psnr:.2f}, SSIM={ssim:.4f}, LPIPS={lpips:.4f}, IFC={ifc:.4f}, KL={kl:.4f}, MI={mi:.4f}, TC={tc:.4f}")
                epoch_results['val_loss'] = val_loss

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_wts = model.state_dict()
                    model_path = os.path.join(models_of_experiment_dir, f"model_epoch_{epoch+1}.pt")
                    # Save the model if the model improved after the current epoch
                    torch.save(best_model_wts, model_path)
                    epochs_no_improve = 0
                    print(f"    > Validation loss improved. Best model updated and saved to {model_path}.")
                else:
                    epochs_no_improve += 1
                    print(f"    > Validation loss did not improve ({epochs_no_improve}/{early_stopping_patience}).")
                    if epochs_no_improve >= early_stopping_patience:
                        print(f"    > Early stopping triggered after {epoch+1} epochs!")
                        break
                
                # Save metrics to CSV file
                write_header = not os.path.exists(metrics_file)  # If file doesn't exist, write header
                with open(metrics_file, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow(["Epoch", "Val Loss", "MSE", "PSNR", "SSIM", "LPIPS", "IFC", "KL", "MI", "TC"])
                    writer.writerow([epoch+1, val_loss, mse, psnr, ssim, lpips, ifc, kl, mi, tc])
            print("//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////\n")

            training_log.append(epoch_results)

            # Visualization of reconstructions
            # results_folder: results/vae_experiment_timestemp
            epoch_results_folder = os.path.join(results_folder, f"epoch_{str(epoch+1).zfill(4)}")
            os.makedirs(epoch_results_folder, exist_ok=True)

            recon_results_folder = os.path.join(epoch_results_folder, "reconstructions")
            os.makedirs(recon_results_folder, exist_ok=True)
            ConvVAE.visualize_reconstructions(
                model=model,
                data_loader=train_loader,
                results_folder=recon_results_folder,
                epoch=epoch,
                num_samples=num_samples_to_show,
                device=device,
                prefix="train_recon",
                title_suffix="(Train Data)"
            )
            if val_loader is not None:
                ConvVAE.visualize_reconstructions(
                    model=model,
                    data_loader=val_loader,
                    results_folder=recon_results_folder,
                    epoch=epoch,
                    num_samples=num_samples_to_show,
                    device=device,
                    prefix="val_recon",
                    title_suffix="(Val Data)"
                )

            # Visualization of anomaly samples (wire images)
            if anomaly_loader is not None:
                epoch_anomaly_folder = os.path.join(epoch_results_folder, "anomaly_wire_analysis")
                os.makedirs(epoch_anomaly_folder, exist_ok=True)
                ConvVAE.visualize_anomalies(
                    model=model,
                    data_loader=anomaly_loader,
                    results_folder=epoch_anomaly_folder,
                    device=device,
                    num_samples=anomaly_samples_per_epoch,
                    threshold=None
                )

        # ---------------- End of Training Loop ---------------- #
        if best_model_wts is not None:
            best_model_path = os.path.join(best_model_dir, model_name)
            # Best model name: "conv_vae.pt"
            torch.save(best_model_wts, best_model_path)
            print(f"\nTraining complete. Best model saved to {best_model_path}")
        else:
            print("\nTraining complete. No validation loader used (or early stopping not triggered).")

        # Save training summary
        results_df = pd.DataFrame(training_log)
        csv_path = os.path.join(results_folder, "training_summary.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"Training summary saved to {csv_path}")

        return model

    @staticmethod
    def validate_model(model, val_loader, device):
        """
        Validates the model on a given validation dataloader.
        """
        # /////////////////////////////////////////////////
        # results folder: results\vae_experiment_08032025_0024
        # /////////////////////////////////////////////////
        model.eval()
        val_loss = 0.0
        evaluator = VAEEvaluator(device)  # Initialize evaluator
        total_mse = 0
        total_psnr = 0
        total_ssim = 0
        total_lpips = 0
        total_ifc = 0
        total_kl = 0
        total_mutual_info = 0
        total_tc = 0
        num_of_samples = len(val_loader.dataset)
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data.to(device)
                batch_size = batch_data.size(0)  # Get batch size
                recon, mu, logvar = model(batch_data)
                # TODO: ASK CHEN ABOUT THE MUKTIPLICATION IN batch_size

                mse_score, psnr_score, ssim_score, lpips_score, ifc_score, kl_div, mutual_info, tc_score = evaluator.evaluate(batch_data, recon, mu, logvar)
                # NO NEED to multiply these by batch_size (they're per image)
                total_mse += mse_score
                total_psnr += psnr_score
                total_ssim += ssim_score
                total_lpips += lpips_score
                total_ifc += ifc_score
                # THESE NEED TO BE MULTIPLIED BY batch_size (they're per batch)
                total_kl += kl_div * batch_size
                total_mutual_info += mutual_info * batch_size
                total_tc += tc_score * batch_size
                
                loss = ConvVAE.loss_function(batch_data, recon, mu, logvar)
                val_loss += loss.item()

        avg_val_loss = val_loss / num_of_samples if num_of_samples > 0 else 0.0
        avg_mse = total_mse/num_of_samples
        avg_psnr = total_psnr/num_of_samples
        avg_ssim = total_ssim/num_of_samples
        avg_lpips = total_lpips/num_of_samples
        avg_ifc = total_ifc/num_of_samples
        avg_kl = total_kl/num_of_samples
        avg_mi = total_mutual_info/num_of_samples
        avg_tc = total_tc/num_of_samples
        return avg_val_loss, avg_mse, avg_psnr, avg_ssim, avg_lpips, avg_ifc, avg_kl, avg_mi, avg_tc

    @staticmethod
    def visualize_reconstructions(model, data_loader, results_folder, epoch, num_samples, device, prefix="recon", title_suffix=""):
        """
        Visualizes input images, their reconstructions, and difference maps.
        """
        model.eval()
        os.makedirs(results_folder, exist_ok=True)

        samples_shown = 0
        with torch.no_grad():
            for batch_data in data_loader:
                batch_data = batch_data.to(device)
                # NEW
                # TODO: change to _,_
                recon, mu, logvar = model(batch_data)

                for i in range(batch_data.size(0)):
                    if samples_shown >= num_samples and num_samples != -1:
                        break

                    orig_img = batch_data[i].cpu().numpy().squeeze()
                    recon_img = recon[i].cpu().numpy().squeeze()
                    
                    # NEW
                    # Compute metrics
                    # mse_score = evaluator.mse(orig_img, recon_img)
                    # psnr_score = evaluator.psnr(orig_img, recon_img)
                    # ssim_score = evaluator.ssim(orig_img, recon_img)
                    # lpips_score = evaluator.lpips(orig_img, recon_img)
                    # ifc_score = evaluator.ifc(orig_img, recon_img)
                    # kl_div = evaluator.kl_divergence(mu, logvar).item()
                    # # mutual_info = evaluator.mutual_information(mu).item()
                    # # tc_score = evaluator.total_correlation(z).item()
                    # mutual_info = evaluator.mutual_information(mu, logvar).item()
                    # tc_score = evaluator.total_correlation(mu, logvar).item()
                    
                    # print(f"Sample {samples_shown+1}: MSE={mse_score:.4f}, PSNR={psnr_score:.2f}, SSIM={ssim_score:.4f}, LPIPS={lpips_score:.4f}, IFC={ifc_score:.4f}, KL={kl_div:.4f}, MI={mutual_info:.4f}, TC={tc_score:.4f}")

                    # Denormalize from [-1, 1] to [0, 1]
                    orig_img_denorm = (orig_img + 1) / 2.0
                    recon_img_denorm = (recon_img + 1) / 2.0

                    orig_img_denorm = np.clip(orig_img_denorm, 0, 1)
                    recon_img_denorm = np.clip(recon_img_denorm, 0, 1)

                    diff_map = np.abs(orig_img_denorm - recon_img_denorm)

                    # Convert difference map to 8-bit for optional thresholding or mask
                    diff_map_8bit = cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

                    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                    axes[0, 0].imshow(orig_img_denorm, cmap='gray')
                    axes[0, 0].set_title("Input")
                    axes[0, 0].axis('off')

                    axes[0, 1].imshow(recon_img_denorm, cmap='gray')
                    axes[0, 1].set_title("Reconstruction")
                    axes[0, 1].axis('off')

                    # Difference heatmap
                    im = axes[1, 0].imshow(diff_map, cmap='viridis', vmin=0, vmax=1)
                    axes[1, 0].set_title("Difference Heatmap")
                    axes[1, 0].axis('off')
                    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

                    # Thresholding for wire overlay
                    _, thresh = cv2.threshold(diff_map_8bit, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    wire_mask = np.ma.masked_where(thresh == 0, thresh)
                    axes[1, 1].imshow(orig_img_denorm, cmap='gray')
                    axes[1, 1].imshow(wire_mask, cmap='autumn', alpha=0.5)
                    axes[1, 1].set_title("Input Overlay")
                    axes[1, 1].axis('off')

                    save_path = os.path.join(
                        results_folder,
                        f"{prefix}_epoch_{str(epoch+1).zfill(4)}_sample_{str(samples_shown+1).zfill(3)}.png"
                    )
                    fig.savefig(save_path, dpi=600)
                    plt.close(fig)
                    samples_shown += 1

                if samples_shown >= num_samples and num_samples != -1:
                    break

    @staticmethod
    def detect_anomalies(model, data_loader, device, threshold=None):
        """
        Returns anomaly scores and (optionally) a boolean classification if a threshold is provided.
        """
        model.eval()
        anomaly_scores = []
        original_images = []
        reconstructed_images = []
        image_paths = []

        for batch_idx, batch_data in enumerate(data_loader):
            batch_data = batch_data.to(device)
            recon, _, _ = model(batch_data)
            batch_mse = F.mse_loss(recon, batch_data, reduction='none').mean(dim=[1, 2, 3])
            anomaly_scores.extend(batch_mse.cpu().detach().numpy())
            original_images.extend(batch_data.cpu().numpy())
            reconstructed_images.extend(recon.cpu().detach().numpy())

            # If the dataset has image_paths, gather them
            if hasattr(data_loader.dataset, 'image_paths'):
                start_index = batch_idx * data_loader.batch_size
                end_index = min(len(data_loader.dataset.image_paths), (batch_idx + 1) * data_loader.batch_size)
                current_batch_paths = data_loader.dataset.image_paths[start_index:end_index]
                image_paths.extend(current_batch_paths)

        anomaly_df = pd.DataFrame({
            'image_path': image_paths,
            'anomaly_score': anomaly_scores
        })

        if threshold is not None:
            anomalies = [score > threshold for score in anomaly_scores]
            anomaly_df['is_anomaly'] = anomalies
            return anomaly_scores, anomalies, original_images, reconstructed_images, anomaly_df
        else:
            return anomaly_scores, original_images, reconstructed_images, anomaly_df

    ########################
    # -- New/Modified Utils
    ########################

    @staticmethod
    def get_discrete_angle(angle, step=15):
        """
        Snaps a continuous angle to the nearest multiple of 'step' degrees.
        """
        return round(angle / step) * step

    @staticmethod
    def define_adaptive_roi_mask(gray_img, walls_mask, default_ratio=0.6):
        """
        Creates an ROI mask by using the bounding box of detected vessel walls.
        If no walls are found, falls back to a fixed top portion of the image.
        """
        mask_roi = np.zeros_like(gray_img, dtype=np.uint8)
        ys, xs = np.where(walls_mask > 0)
        if len(ys) > 0:
            y_min, y_max = np.min(ys), np.max(ys)
            # Expand slightly or clamp if desired
            y_min = max(0, y_min - 5)
            y_max = min(gray_img.shape[0], y_max + 5)
            mask_roi[y_min:y_max, :] = 255
        else:
            h = gray_img.shape[0]
            roi_height = int(h * default_ratio)
            mask_roi[:roi_height, :] = 255
        return mask_roi

    @staticmethod
    def calculate_geometric_features(contour):
        """
        Calculates geometric features for a contour.
        Uses discrete 15° angle steps.
        
        Guards against contours with fewer than 5 points, 
        which are invalid for cv2.fitEllipse.
        """
        if len(contour) < 5:
            # Return default/placeholder values if not enough points
            return 0, 0, 0, 0

        ellipse = cv2.fitEllipse(contour)
        continuous_angle = ellipse[-1]
        # Snap angle to nearest multiple of 15 degrees
        snapped_angle = ConvVAE.get_discrete_angle(continuous_angle, step=15)

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        elongation = 1 - (float(h) / w) if w > 0 else 0
        area = cv2.contourArea(contour)
        return snapped_angle, aspect_ratio, elongation, area

    @staticmethod
    def calculate_intensity_features(gray_img, contour):
        """
        Calculates intensity-based features for a contour.
        Includes mean intensity, percentile stats, local contrast (std), and variance.
        """
        mask = np.zeros_like(gray_img, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        mean_intensity = cv2.mean(gray_img, mask=mask)[0]

        pixels_in_contour = gray_img[mask > 0]
        percentile_75 = np.percentile(pixels_in_contour, 75) if pixels_in_contour.size > 0 else 0
        percentile_90 = np.percentile(pixels_in_contour, 90) if pixels_in_contour.size > 0 else 0

        local_contrast_ratio = np.std(pixels_in_contour) if pixels_in_contour.size > 1 else 0
        pixel_variance = np.var(pixels_in_contour) if pixels_in_contour.size > 1 else 0

        return mean_intensity, percentile_75, percentile_90, local_contrast_ratio, pixel_variance

    @staticmethod
    def detect_vessel_walls(gray_img, roi_mask, wall_angle_range=[-60, 60], wall_intensity_threshold=150):
        """
        Simplified vessel wall detection using Canny + Probabilistic Hough transform.
        """
        masked_img = cv2.bitwise_and(gray_img, gray_img, mask=roi_mask) if roi_mask is not None else gray_img
        edges = cv2.Canny(masked_img, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)
        walls_mask = np.zeros_like(gray_img, dtype=np.uint8)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Check angle range (accounting for ±180 wrap-around if needed)
                if (wall_angle_range[0] <= angle <= wall_angle_range[1]) or (wall_angle_range[0] <= angle + 180 <= wall_angle_range[1]):
                    avg_intensity = (int(gray_img[y1, x1]) + int(gray_img[y2, x2])) / 2
                    if avg_intensity > wall_intensity_threshold:
                        cv2.line(walls_mask, (x1, y1), (x2, y2), 255, 2)

        return walls_mask

    @staticmethod
    def calculate_anatomical_context_features(contour, walls_mask, inter_wall_spacing_ratio_target=0.1):
        """
        Calculates anatomical context features based on vessel walls.
        For now, we compute minimal distance from wire centroid to any wall pixel,
        normalized by image height. Inter-wall spacing is a placeholder.
        """
        contour_mask = np.zeros_like(walls_mask, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        wire_pixels_y, wire_pixels_x = np.where(contour_mask > 0)

        if not wire_pixels_x.size:
            return 0, 0  # no pixels

        wire_centroid_x = np.mean(wire_pixels_x)
        wire_centroid_y = np.mean(wire_pixels_y)

        wall_pixels_y, wall_pixels_x = np.where(walls_mask > 0)
        if not wall_pixels_x.size:
            return 0, 0

        # Euclidean distance from wire centroid to all wall pixels
        dist_x = wall_pixels_x - wire_centroid_x
        dist_y = wall_pixels_y - wire_centroid_y
        distances = np.sqrt(dist_x**2 + dist_y**2)

        min_distance_to_wall = np.min(distances) if distances.size > 0 else 0
        image_height = walls_mask.shape[0]
        wire_to_wall_distance_ratio = min_distance_to_wall / image_height if image_height > 0 else 0

        # Placeholder for inter-wall spacing
        inter_wall_spacing_consistency = 0

        return wire_to_wall_distance_ratio, inter_wall_spacing_consistency

    @staticmethod
    def score_wire_candidate(geometric_score, intensity_score, anatomical_score, w1=0.4, w2=0.3, w3=0.3):
        """
        Scores a wire candidate based on geometric, intensity, and anatomical scores.
        """
        return w1 * geometric_score + w2 * intensity_score + w3 * anatomical_score

    @staticmethod
    def visualize_anomalies(
        model, 
        data_loader, 
        results_folder, 
        device, 
        num_samples=-1, 
        threshold=None,
        # Parameter Ranges
        angle_range=[-60, 60],
        roi_height_ratio_range=[0.4, 0.8],
        aspect_ratio_range=[3.0, 7.0],
        elongation_threshold=0.7,
        intensity_threshold_ratio_range=[1.2, 2.0],
        percentile_range=[75, 90],
        local_contrast_threshold=10.0,
        adaptive_block_size_ratio_range=[0.02, 0.06],
        adaptive_c_range=[1, 5],
        wall_spacing_ratio_range=[0.05, 0.15], 
        wall_intensity_threshold=150, 
        wall_angle_range=[-60, 60],
        inter_wall_spacing_ratio_target=0.1,
        min_size_ratio_range=[0.0003, 0.0007],
        max_size_ratio_range=[0.01, 0.02],
        mean_intensity_threshold_range=[0.6, 0.9],
        variance_threshold=50.0,
        opening_kernels=[3, 5, 7],
        dilation_iterations_options=[2, 3, 4]
    ):
        """
        Visualizes anomaly detection pipeline on selected samples.
        Incorporates random parameter sampling for demonstration.
        """
        model.eval()
        os.makedirs(results_folder, exist_ok=True)

        # Detect anomalies (get MSE scores) but do not threshold globally
        anomaly_scores, original_images, reconstructed_images, _ = ConvVAE.detect_anomalies(
            model, data_loader, device, threshold
        )

        # Decide how many samples to show
        if 0 < num_samples < len(original_images):
            sampled_indices = random.sample(range(len(original_images)), num_samples)
        else:
            sampled_indices = range(len(original_images))

        samples_shown = 0
        for i in sampled_indices:
            if samples_shown == num_samples and num_samples != -1:
                break

            orig_img = original_images[i].squeeze()
            recon_img = reconstructed_images[i].squeeze()
            score = anomaly_scores[i]

            # Denormalize [-1,1] -> [0,1]
            orig_img_denorm = np.clip((orig_img + 1) / 2.0, 0, 1)
            recon_img_denorm = np.clip((recon_img + 1) / 2.0, 0, 1)

            orig_img_8bit = (orig_img_denorm * 255).astype(np.uint8)
            heatmap = np.abs(orig_img_denorm - recon_img_denorm)
            heatmap_8bit = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

            # -----------------------------
            # Randomly pick pipeline params
            # -----------------------------
            current_roi_height_ratio = random.uniform(*roi_height_ratio_range)
            current_adaptive_block_size_ratio = random.uniform(*adaptive_block_size_ratio_range)
            current_adaptive_c = random.randint(*adaptive_c_range)

            current_min_size_ratio = random.uniform(*min_size_ratio_range)
            current_max_size_ratio = random.uniform(*max_size_ratio_range)

            # Example: step through aspect ratios in [3.0..7.0] by 1.0, pick randomly
            possible_aspect_ratios = [3.0, 4.0, 5.0, 6.0, 7.0]
            current_aspect_ratio_threshold = random.choice(possible_aspect_ratios)

            current_elongation_threshold = elongation_threshold

            current_opening_kernel_size = random.choice(opening_kernels)
            current_dilation_iterations = random.choice(dilation_iterations_options)

            # Example: pick an intensity threshold ratio from [1.2..2.0] in 0.2 steps
            possible_int_ratios = [round(x, 1) for x in np.arange(1.2, 2.1, 0.2)]
            current_intensity_threshold_ratio = random.choice(possible_int_ratios)

            current_local_contrast_threshold = local_contrast_threshold
            current_wall_intensity_threshold = wall_intensity_threshold
            current_wall_angle_range = wall_angle_range
            current_inter_wall_spacing_ratio_target = inter_wall_spacing_ratio_target

            # ------------------------------------------------------------------------
            # Begin anomaly-detection pipeline (with morphological and ROI adaptivity)
            # ------------------------------------------------------------------------

            # 1) Initial vessel wall detection (no ROI restriction yet).
            full_mask = np.ones_like(orig_img_8bit, dtype=np.uint8) * 255
            initial_walls_mask = ConvVAE.detect_vessel_walls(
                orig_img_8bit, 
                roi_mask=full_mask, 
                wall_angle_range=current_wall_angle_range, 
                wall_intensity_threshold=current_wall_intensity_threshold
            )

            # 2) Define ROI adaptively from vessel walls bounding box
            adaptive_roi_mask = ConvVAE.define_adaptive_roi_mask(orig_img_8bit, initial_walls_mask, default_ratio=current_roi_height_ratio)

            # 3) Threshold the difference map only in the ROI
            roi_heatmap = cv2.bitwise_and(heatmap_8bit, heatmap_8bit, mask=adaptive_roi_mask)

            block_size = int(orig_img_8bit.shape[0] * current_adaptive_block_size_ratio)
            if block_size % 2 == 0:
                block_size += 1  # Must be odd for OpenCV adaptiveThreshold

            thresh = cv2.adaptiveThreshold(
                roi_heatmap, 
                255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                block_size, 
                current_adaptive_c
            )

            # 4) Morphological cleaning
            #    - Horizontal opening kernel
            opening_kernel = np.ones((current_opening_kernel_size, 1), np.uint8)
            opened_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, opening_kernel)

            #    - Dilation with rectangular kernel
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
            dilated_thresh = cv2.dilate(opened_thresh, horizontal_kernel, iterations=current_dilation_iterations)
            final_bin = dilated_thresh

            # 5) Connected components (contour detection)
            contours, hierarchy = cv2.findContours(final_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            wire_mask = np.zeros_like(final_bin)
            final_contours = []

            # 6) Vessel walls for final pass (already have initial_walls_mask).
            #    Potentially re-run with ROI if desired, but here we'll keep the same walls.
            walls_mask = initial_walls_mask

            image_mean_intensity = np.mean(orig_img_8bit)
            height, width = orig_img_8bit.shape[:2]

            # Evaluate each contour
            for contour in contours:
                angle, aspect_ratio, elongation, area = ConvVAE.calculate_geometric_features(contour)
                (mean_intensity, p75, p90, local_contrast_ratio, pixel_variance
                 ) = ConvVAE.calculate_intensity_features(orig_img_8bit, contour)
                (wire_to_wall_distance_ratio, inter_wall_spacing_consistency
                 ) = ConvVAE.calculate_anatomical_context_features(
                    contour, walls_mask, inter_wall_spacing_ratio_target=current_inter_wall_spacing_ratio_target
                )

                # If the contour is too small (ellipse defaults came back 0, 0, 0, 0), skip
                if area == 0:
                    continue

                # Size check
                min_area = current_min_size_ratio * (width * height)
                max_area = current_max_size_ratio * (width * height)
                size_valid = (min_area < area < max_area)

                # Angle check
                angle_valid = (angle_range[0] <= angle <= angle_range[1])

                # Aspect & elongation
                aspect_ratio_valid = (aspect_ratio > current_aspect_ratio_threshold)
                elongation_valid = (elongation > current_elongation_threshold)

                # Intensity-based checks
                intensity_threshold_valid = (mean_intensity > current_intensity_threshold_ratio * image_mean_intensity)
                local_contrast_valid = (local_contrast_ratio > current_local_contrast_threshold)

                # Variance check (for uniform wire region)
                variance_valid = (pixel_variance < variance_threshold)

                # Composite sub-scores (binary for demonstration)
                geometric_score = 1.0 if (size_valid and angle_valid and aspect_ratio_valid and elongation_valid) else 0.0
                intensity_score = 1.0 if (intensity_threshold_valid and local_contrast_valid and variance_valid) else 0.0
                anatomical_score = 1.0 if (wire_to_wall_distance_ratio > 0) else 0.0  # placeholder

                final_score = ConvVAE.score_wire_candidate(geometric_score, intensity_score, anatomical_score)

                # Example threshold on final_score
                if final_score > 0.5:
                    final_contours.append(contour)

            # Draw final contours
            cv2.drawContours(wire_mask, final_contours, -1, 255, thickness=cv2.FILLED)
            wire_mask_ma = np.ma.masked_where(wire_mask == 0, wire_mask)

            # -----------------
            # Visualization
            # -----------------
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))

            # 1) Original input
            axes[0].imshow(orig_img_denorm, cmap='gray')
            axes[0].set_title("Input (Wire)")
            axes[0].axis('off')

            # 2) Overlaid final contours
            axes[1].imshow(orig_img_denorm, cmap='gray')
            im = axes[1].imshow(wire_mask_ma, cmap='magma', alpha=0.5, interpolation='nearest')
            axes[1].set_title(f"Wire Overlay - Score: {score:.4f}")
            axes[1].axis('off')
            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

            # 3) Vessel walls overlay
            axes[2].imshow(orig_img_8bit, cmap='gray')
            axes[2].imshow(walls_mask, cmap='autumn', alpha=0.5)
            axes[2].set_title("Vessel Walls (Adaptive ROI)")
            axes[2].axis('off')

            save_path = os.path.join(
                results_folder,
                f"anomaly_sample_{str(samples_shown+1).zfill(3)}_score_{score:.4f}.png"
            )
            fig.savefig(save_path, dpi=600)
            plt.close(fig)

            samples_shown += 1

###############################################################################
# Main script entry point
###############################################################################

def main():
    # Define transforms
    custom_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((DEFAULT_RESIZE_HEIGHT, DEFAULT_RESIZE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    root_dir = ROOT_DATA_DIR
    config_file = CONFIG_FILE_NAME
    resize_height = DEFAULT_RESIZE_HEIGHT
    resize_width = DEFAULT_RESIZE_WIDTH
    max_training_images = DEFAULT_MAX_TRAINING_IMAGES
    batch_size = DEFAULT_BATCH_SIZE
    num_workers = 4 if torch.cuda.is_available() else 0
    pin_memory = True if torch.cuda.is_available() else False

    latent_dim = DEFAULT_LATENT_DIM
    num_epochs = DEFAULT_NUM_EPOCHS
    learning_rate = DEFAULT_LEARNING_RATE
    step_size = DEFAULT_STEP_SIZE
    gamma = DEFAULT_GAMMA
    model_name = DEFAULT_MODEL_NAME
    early_stopping_patience = EARLY_STOPPING_PATIENCE
    anomaly_samples_per_epoch = ANOMALY_VISUALIZATION_SAMPLES_PER_EPOCH

    # For train/val split
    val_split_ratio = 0.2

    # timestamp = datetime.now().strftime("%d%m%Y_%H%M")
    # results_folder_name = f"vae_experiment_{timestamp}"
    results_folder = os.path.join(RESULTS_DIR, RESULTS_FOLDER_NAME)
    os.makedirs(results_folder, exist_ok=True)

    anomaly_results_folder = os.path.join(ANOMALY_RESULTS_DIR, RESULTS_FOLDER_NAME)
    os.makedirs(anomaly_results_folder, exist_ok=True)

    # Create datasets
    full_dataset, anomaly_dataset = create_datasets()

    # Split train/val
    val_size = int(val_split_ratio * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print("Train set size:", len(train_dataset))
    print("Val set size:", len(val_dataset))

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    anomaly_loader = DataLoader(
        anomaly_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Train VAE
    vae_model = ConvVAE.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        anomaly_loader=anomaly_loader,
        input_height=resize_height,
        input_width=resize_width,
        latent_dim=latent_dim,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        step_size=step_size,
        gamma=gamma,
        model_name=model_name,
        device=device,
        num_samples_to_show=DEFAULT_NUM_SAMPLES_TO_SHOW,
        results_folder=results_folder,
        early_stopping_patience=early_stopping_patience,
        anomaly_samples_per_epoch=anomaly_samples_per_epoch
    )

    # Final anomaly analysis
    final_anomaly_results_folder = os.path.join(anomaly_results_folder, "final_analysis")
    os.makedirs(final_anomaly_results_folder, exist_ok=True)

    anomaly_scores, _, _, anomaly_df = ConvVAE.detect_anomalies(vae_model, anomaly_loader, device=device)

    # Visualize anomalies with the improved parameter pipeline
    ConvVAE.visualize_anomalies(
        model=vae_model,
        data_loader=anomaly_loader,
        results_folder=final_anomaly_results_folder,
        device=device,
        num_samples=-1,  # Show all anomaly samples
        threshold=None,
        # Parameter Ranges (now subject to random sampling in the code)
        angle_range=[-60, 60],
        roi_height_ratio_range=[0.4, 0.8],
        aspect_ratio_range=[3.0, 7.0],
        elongation_threshold=0.7,
        intensity_threshold_ratio_range=[1.2, 2.0],
        percentile_range=[75, 90],
        local_contrast_threshold=10.0,
        adaptive_block_size_ratio_range=[0.02, 0.06],
        adaptive_c_range=[1, 5],
        wall_spacing_ratio_range=[0.05, 0.15],
        wall_intensity_threshold=150,
        wall_angle_range=[-60, 60],
        inter_wall_spacing_ratio_target=0.1,
        min_size_ratio_range=[0.0003, 0.0007],
        max_size_ratio_range=[0.01, 0.02],
        mean_intensity_threshold_range=[0.6, 0.9],
        variance_threshold=50.0,
        opening_kernels=[3, 5, 7],
        dilation_iterations_options=[2, 3, 4]
    )

    # Save final anomaly summary
    anomaly_csv_path = os.path.join(final_anomaly_results_folder, "anomaly_summary.csv")
    anomaly_df.to_csv(anomaly_csv_path, index=False)
    print(f"Final anomaly summary saved to {anomaly_csv_path}")

    print("Done.")

if __name__ == "__main__":
    main()