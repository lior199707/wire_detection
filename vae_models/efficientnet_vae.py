import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
import os

class EfficientNetVAE(Model):
    MODEL_DIR = "models"
    def __init__(self, latent_dim=512):
        super(EfficientNetVAE, self).__init__()

        # Encoder - EfficientNetB0
        base_model = EfficientNetB0(include_top=False, input_shape=(512, 1024, 1), weights=None)
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(512, 1024, 1)),
            layers.Conv2D(3, (1, 1), activation='relu'),  # Convert grayscale to RGB
            base_model,
            layers.Flatten(),
            layers.Dense(latent_dim * 2)  # Output mu and logvar
        ])

        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Dense(16 * 32 * 512, activation="relu"),
            layers.Reshape((16, 32, 512)),
            layers.Conv2DTranspose(256, (3, 3), strides=2, padding='same', activation="relu"),
            layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation="relu"),
            layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation="relu"),
            layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation="relu"),
            layers.Conv2DTranspose(1, (3, 3), strides=1, padding='same', activation="sigmoid")  # Output grayscale
        ])

    def reparameterize(self, mu, logvar):
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(0.5 * logvar) * eps

    def call(self, x):
        enc_output = self.encoder(x)
        mu, logvar = tf.split(enc_output, num_or_size_splits=2, axis=1)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    @staticmethod
    def vae_loss(x, recon_x, mu, logvar):
        recon_loss = tf.reduce_sum(tf.keras.losses.MeanSquaredError()(x, recon_x))
        kl_div = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar))
        return recon_loss + kl_div

    @staticmethod
    def train_step(vae, images, optimizer):
        with tf.GradientTape() as tape:
            recon_images, mu, logvar = vae(images)
            loss = EfficientNetVAE.vae_loss(images, recon_images, mu, logvar)

        gradients = tape.gradient(loss, vae.trainable_variables)
        optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
        return loss

    @staticmethod
    def train_model(train_data, num_epochs=50, latent_dim=512, learning_rate=1e-4, model_name="efficientnet_vae_ultrasound.h5"):
        # Initialize model
        vae = EfficientNetVAE(latent_dim=latent_dim)

        # Optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_data:  # Replace 'train_data' with your actual dataset
                loss = EfficientNetVAE.train_step(vae, batch, optimizer)
                total_loss += loss.numpy()
                break

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_data):.4f}")

        # Save model weights
        os.makedirs(EfficientNetVAE.MODEL_DIR, exist_ok=True)
        vae.save_weights(os.path.join(EfficientNetVAE.MODEL_DIR, model_name))

    @staticmethod
    def load_model(latent_dim=512, model_name="efficientnet_vae_ultrasound.h5"):
        vae = EfficientNetVAE(latent_dim=latent_dim)
        vae.load_weights(os.path.join(EfficientNetVAE.MODEL_DIR, model_name))
        return vae