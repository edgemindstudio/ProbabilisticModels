import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tqdm import trange
from tensorflow.keras.optimizers.legacy import Adam
from scipy.linalg import sqrtm
import datetime
import tensorflow_probability as tfp

# -------------------------
# Config
# -------------------------
IMG_SHAPE = (40, 40, 1)
LATENT_DIM = 100
BATCH_SIZE = 256
EPOCHS = 10000
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "USTC-TFC2016_malware")
os.makedirs("dcgan_checkpoints", exist_ok=True)

# -------------------------
# Load & Normalize Data [-1, 1] for tanh
# -------------------------
train_data = np.load(os.path.join(DATA_PATH, "train_data.npy"), allow_pickle=True)
train_data = train_data.reshape(-1, *IMG_SHAPE)
train_data = (train_data - 0.5) * 2.0

# -------------------------
# Load InceptionV3 model for FID
# -------------------------
inception_model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(75, 75, 3))

# -------------------------
# FID Calculation
# -------------------------
def calculate_fid(real_images, generated_images):

    # Ensure InceptionV3 gets valid images | Rescale [-1, 1] -> [0, 255]
    generated_images = (generated_images + 1) * 127.5
    real_images = (real_images + 1) * 127.5

    real_images_resized = tf.image.resize(real_images, (75, 75))
    fake_images_resized = tf.image.resize(generated_images, (75, 75))

    real_images_resized = tf.repeat(real_images_resized, repeats=3, axis=-1)
    fake_images_resized = tf.repeat(fake_images_resized, repeats=3, axis=-1)

    real_activations = inception_model(real_images_resized)
    fake_activations = inception_model(fake_images_resized)

    mu1 = tf.reduce_mean(real_activations, axis=0)
    mu2 = tf.reduce_mean(fake_activations, axis=0)
    sigma1 = tfp.stats.covariance(real_activations)
    sigma2 = tfp.stats.covariance(fake_activations)

    ssdiff = tf.reduce_sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    if np.isnan(covmean).any():
        print("WARNING: NaN in FID sqrtm! Skipping.")
        return 1e6

    fid = ssdiff + tf.linalg.trace(sigma1 + sigma2 - 2 * covmean)
    return fid.numpy()

# -------------------------
# DCGAN Generator
# -------------------------
def build_generator():
    model = models.Sequential([
        layers.Input(shape=(LATENT_DIM,)),
        layers.Dense(5 * 5 * 256, use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Reshape((5, 5, 256)),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding='same', activation='tanh', use_bias=False)
    ])

    model.summary()
    return model

# -------------------------
# DCGAN Discriminator
# -------------------------
def build_discriminator():
    model = models.Sequential([
        layers.Input(shape=IMG_SHAPE),
        layers.Conv2D(64, kernel_size=4, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Conv2D(128, kernel_size=4, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# -------------------------
# Helper: Save Samples
# -------------------------
def save_samples(generator, epoch, n=10):
    noise = np.random.normal(0, 1, (n, LATENT_DIM))
    generated_samples = generator.predict(noise)

    # Rescale from [-1, 1] → [0, 255]
    generated_samples = np.clip((generated_samples + 1) * 127.5, 0, 255).astype(np.uint8)

    # Save same format as simple GAN
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
    for i in range(n):
        axes[i].imshow(generated_samples[i].reshape(40, 40), cmap='gray')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(f"generated_sample_epoch_{epoch}.png")
    plt.close()

# -------------------------
# Training Loop
# -------------------------
def train_dcgan():
    generator = build_generator()
    discriminator = build_discriminator()

    discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])
    discriminator.trainable = False

    z = tf.keras.Input(shape=(LATENT_DIM,))
    img = generator(z)
    valid = discriminator(img)
    gan = tf.keras.Model(z, valid)
    gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

    d_losses, g_losses, fid_scores = [], [], []

    for epoch in trange(EPOCHS):
        idx = np.random.randint(0, train_data.shape[0], BATCH_SIZE)
        real_imgs = train_data[idx]

        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        fake_imgs = generator.predict(noise)

        # Debug pixel range
        if epoch % 1000 == 0:
            print("Fake sample pixel range:", np.min(fake_imgs), np.max(fake_imgs))

        # Label smoothing
        real_labels = np.random.uniform(0.85, 1.0, size=(BATCH_SIZE, 1))
        fake_labels = np.random.uniform(0.0, 0.15, size=(BATCH_SIZE, 1))

        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(real_imgs, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        discriminator.trainable = False
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        g_loss = gan.train_on_batch(noise, np.ones((BATCH_SIZE, 1)))

        d_losses.append(d_loss[0])
        g_losses.append(g_loss)

        if epoch % 1000 == 0:
            save_samples(generator, epoch)
            print(f"Epoch {epoch} | D Loss: {d_loss[0]:.4f}, G Loss: {g_loss:.4f}, D Acc: {d_loss[1]*100:.2f}%")

            fid = calculate_fid(real_imgs[:100], generator.predict(np.random.normal(0, 1, (100, LATENT_DIM))))
            fid_scores.append(fid)
            print(f"FID Score at Epoch {epoch}: {fid:.2f}")

            generator.save_weights(f"dcgan_checkpoints/generator_epoch_{epoch}.h5")
            discriminator.save_weights(f"dcgan_checkpoints/discriminator_epoch_{epoch}.h5")

    # Plot and save loss & FID curves
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('DCGAN Training Loss')
    plt.grid(True)
    plt.savefig("dcgan_loss_plot.png")
    plt.close()

    plt.plot(fid_scores, label='FID Score')
    plt.xlabel('Checkpoint (every 1000 epochs)')
    plt.ylabel('FID')
    plt.title('FID over Epochs')
    plt.grid(True)
    plt.legend()
    plt.savefig("dcgan_fid_plot.png")
    plt.close()

# -------------------------
# Run Training
# -------------------------
if __name__ == "__main__":
    train_dcgan()
