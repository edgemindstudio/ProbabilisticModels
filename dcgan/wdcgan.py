import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tqdm import trange
from tensorflow.keras.optimizers.legacy import RMSprop
from scipy.linalg import sqrtm
import datetime
import tensorflow_probability as tfp

# -------------------------
# Config
# -------------------------
IMG_SHAPE = (40, 40, 1)
LATENT_DIM = 100
BATCH_SIZE = 64
EPOCHS = 10000
N_CRITIC = 5
LAMBDA_GP = 10.0
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "USTC-TFC2016_malware")
os.makedirs("wgan_gp_checkpoints", exist_ok=True)

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
    real_images = (real_images + 1) * 127.5
    generated_images = (generated_images + 1) * 127.5

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
        print("WARNING: NaN detected in sqrtm during FID calculation")
        return 1e6
    fid = ssdiff + tf.linalg.trace(sigma1 + sigma2 - 2 * covmean)
    return fid.numpy()

# -------------------------
# Generator
# -------------------------
def build_generator():
    model = models.Sequential([
        layers.Input(shape=(LATENT_DIM,)),
        layers.Dense(5 * 5 * 256, use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),

        layers.Reshape((5, 5, 256)),
        layers.Conv2DTranspose(128, 4, 2, 'same', use_bias=False),
        layers.BatchNormalization(), layers.LeakyReLU(0.2),

        layers.Conv2DTranspose(64, 4, 2, 'same', use_bias=False),
        layers.BatchNormalization(), layers.LeakyReLU(0.2),
        
        layers.Conv2DTranspose(1, 4, 2, 'same', activation='tanh', use_bias=False)
    ])
    return model

# -------------------------
# Discriminator (No sigmoid)
# -------------------------
def build_discriminator():
    model = models.Sequential([
        layers.Input(shape=IMG_SHAPE),
        layers.Conv2D(64, 4, 2, 'same'), layers.LeakyReLU(0.2), layers.Dropout(0.3),
        layers.Conv2D(128, 4, 2, 'same'), layers.LeakyReLU(0.2), layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)  # No activation
    ])
    return model

# -------------------------
# Gradient Penalty
# -------------------------
def gradient_penalty(discriminator, real, fake):
    alpha = tf.random.uniform([real.shape[0], 1, 1, 1], 0.0, 1.0)
    interpolated = alpha * real + (1 - alpha) * fake
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator(interpolated)
    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-10)
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

# -------------------------
# Training
# -------------------------
def train_wgan_gp():
    generator = build_generator()
    discriminator = build_discriminator()
    g_optimizer = RMSprop(learning_rate=0.00005)
    d_optimizer = RMSprop(learning_rate=0.00005)

    d_losses, g_losses, fid_scores = [], [], []

    for epoch in trange(EPOCHS):
        for _ in range(N_CRITIC):
            idx = np.random.randint(0, train_data.shape[0], BATCH_SIZE)
            real_imgs = train_data[idx]
            noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
            fake_imgs = generator(noise, training=True)

            with tf.GradientTape() as d_tape:
                real_validity = discriminator(real_imgs, training=True)
                fake_validity = discriminator(fake_imgs, training=True)
                gp = gradient_penalty(discriminator, real_imgs, fake_imgs)
                d_loss = tf.reduce_mean(fake_validity) - tf.reduce_mean(real_validity) + LAMBDA_GP * gp

            d_grad = d_tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(d_grad, discriminator.trainable_variables))

        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        with tf.GradientTape() as g_tape:
            gen_imgs = generator(noise, training=True)
            g_loss = -tf.reduce_mean(discriminator(gen_imgs, training=True))

        g_grad = g_tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(g_grad, generator.trainable_variables))

        d_losses.append(d_loss.numpy())
        g_losses.append(g_loss.numpy())

        if epoch % 1000 == 0:
            print(f"Epoch {epoch} | D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")
            noise_sample = np.random.normal(0, 1, (10, LATENT_DIM))
            gen_samples = generator(noise_sample, training=False)
            gen_imgs = ((gen_samples + 1) * 127.5).numpy().astype(np.uint8)
            fig, axes = plt.subplots(1, 10, figsize=(20, 2))
            for i in range(10):
                axes[i].imshow(gen_imgs[i].reshape(40, 40), cmap='gray')
                axes[i].axis('off')
            plt.tight_layout()
            plt.savefig(f"wgan_gp_checkpoints/generated_epoch_{epoch}.png")
            plt.close()

            fid = calculate_fid(train_data[:100], gen_samples[:100])
            fid_scores.append(fid)
            print(f"FID Score at Epoch {epoch}: {fid:.2f}")

            generator.save_weights(f"wgan_gp_checkpoints/generator_epoch_{epoch}.h5")
            discriminator.save_weights(f"wgan_gp_checkpoints/discriminator_epoch_{epoch}.h5")

    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('WGAN-GP Loss Curves')
    plt.grid(True)
    plt.savefig("wgan_gp_loss_plot.png")
    plt.close()

    plt.plot(fid_scores, label='FID Score')
    plt.xlabel('Checkpoint (every 1000 epochs)')
    plt.ylabel('FID')
    plt.legend()
    plt.title('WGAN-GP FID over Epochs')
    plt.grid(True)
    plt.savefig("wgan_gp_fid_plot.png")
    plt.close()

# -------------------------
# Run Training
# -------------------------
if __name__ == "__main__":
    train_wgan_gp()