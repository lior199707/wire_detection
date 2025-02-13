import os
from dataset_loader.image_database import ImageDatabase
from dataset_loader.ultrasound_dataset import UltrasoundDataset
from vae_models import UNetVAE, EfficientNetVAE
from torch.utils.data import DataLoader
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array


if __name__ == '__main__':
    # Define paths
    root_dir = os.path.join("data")
    config_file = "dataset_config.json"

    dataset = UltrasoundDataset(root_dir, config_file, "no_wire")

    # unet vae
    #//////////////////////////////////////////////////////////////////////////////////
    # # TODO: unet vae - model creation
    #
    # print(len(dataset.image_paths))
    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    # UNetVAE.train_model(dataloader, num_epochs=1)
    #
    # # TODO: unet vae - image reconstruction
    # import torch
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # unet_vae = UNetVAE.load_model()
    # dataset = UltrasoundDataset(root_dir, config_file, "wire")
    # # Get a test image
    # test_img = dataset[0].unsqueeze(0).to(device)
    #
    # # Reconstruct
    # with torch.no_grad():
    #     recon_img, _, _ = unet_vae(test_img)
    #
    # # Convert to NumPy
    # test_img = test_img.cpu().squeeze().numpy()
    # recon_img = recon_img.cpu().squeeze().numpy()
    #
    # # Display
    # plt.subplot(1, 2, 1)
    # plt.title("Original")
    # plt.imshow(test_img, cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.title("Reconstructed")
    # plt.imshow(recon_img, cmap="gray")
    # plt.show()
    # //////////////////////////////////////////////////////////////////////////////////




    # Efficient vae image reconstruction
    # //////////////////////////////////////////////////////////////////////////////////
    # TODO: efficient vae - model creation
    images = []
    for path in dataset.get_image_paths():
        img = load_img(path, color_mode="grayscale", target_size=(512, 1024))
        img_array = img_to_array(img) / 255.0  # Normalize
        images.append(img_array)

    image_data = np.array(images)
    train_data = tf.data.Dataset.from_tensor_slices(image_data).shuffle(100).batch(8)
    EfficientNetVAE.train_model(train_data, num_epochs=1)

    # TODO: efficient vae - image reconstruction
    # efficient_vae = EfficientNetVAE.load_model()
    # # Get a test image
    # test_img = image_data[0:1]  # Single test image
    #
    # # Reconstruct
    # recon_img, _, _ = efficient_vae(test_img)
    #
    # # Convert to NumPy
    # test_img = np.squeeze(test_img)
    # recon_img = np.squeeze(recon_img)
    #
    # # Display
    # plt.subplot(1, 2, 1)
    # plt.title("Original")
    # plt.imshow(test_img, cmap="gray")
    # plt.subplot(1, 2, 2)
    # plt.title("Reconstructed")
    # plt.imshow(recon_img, cmap="gray")
    # plt.show()
    # //////////////////////////////////////////////////////////////////////////////////



