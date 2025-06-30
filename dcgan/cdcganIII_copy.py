import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.linalg import sqrtm
import tensorflow_probability as tfp
from datetime import datetime
import sys

LOG_FILE = "conditional_dcgan_result.txt"

def log_message(message, display=True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    if display:
        print(full_msg)
    with open(LOG_FILE, "a") as f:
        f.write(full_msg + "\n")

# -------------------------------
# Config (Hyperparameters)
# -------------------------------
IMG_SHAPE = (40, 40, 1)
LATENT_DIM = 100
NUM_CLASSES = 9
BATCH_SIZE = 256
EPOCHS = 10000
LR = 0.0002
BETA_1 = 0.5
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "USTC-TFC2016_malware")
LOG_DIR = "logs"
TENSORBOARD_DIR = os.path.join(LOG_DIR, "tensorboard", datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs("cdgan_checkpoints", exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Save hyperparameters
with open(os.path.join(LOG_DIR, "hyperparameters.txt"), "w") as f:
    f.write(f"IMG_SHAPE={IMG_SHAPE}, LATENT_DIM={LATENT_DIM}, NUM_CLASSES={NUM_CLASSES}, \
            BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, LR={LR}, BETA_1={BETA_1}\n")

# TensorBoard writer
summary_writer = tf.summary.create_file_writer(TENSORBOARD_DIR)

# -------------------------------
# Load & Normalize Data
# -------------------------------
train_data = np.load(os.path.join(DATA_PATH, "train_data.npy"), allow_pickle=True)
train_labels = np.load(os.path.join(DATA_PATH, "train_labels.npy"), allow_pickle=True)
train_data = train_data.reshape(-1, *IMG_SHAPE)
train_data = (train_data - 0.5) * 2.0
train_labels = tf.keras.utils.to_categorical(train_labels, NUM_CLASSES)

# -------------------------------
# FID Calculation
# -------------------------------
inception_model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(75, 75, 3))


def calculate_fid(real_images, generated_images):
    real_images = (real_images + 1) * 127.5
    generated_images = (generated_images + 1) * 127.5
    real_images_resized = tf.image.resize(real_images, (75, 75))
    fake_images_resized = tf.image.resize(generated_images, (75, 75))
    real_images_resized = tf.repeat(real_images_resized, 3, -1)
    fake_images_resized = tf.repeat(fake_images_resized, 3, -1)
    real_activations = inception_model(real_images_resized)
    fake_activations = inception_model(fake_images_resized)
    mu1 = tf.reduce_mean(real_activations, axis=0)
    mu2 = tf.reduce_mean(fake_activations, axis=0)
    sigma1 = tfp.stats.covariance(real_activations)
    sigma2 = tfp.stats.covariance(fake_activations)
    ssdiff = tf.reduce_sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean): covmean = covmean.real
    if np.isnan(covmean).any(): return 1e6
    return (ssdiff + tf.linalg.trace(sigma1 + sigma2 - 2 * covmean)).numpy()


# -------------------------------
# Models
# -------------------------------
def build_generator():
    noise_input = layers.Input(shape=(LATENT_DIM,))
    label_input = layers.Input(shape=(NUM_CLASSES,))
    x = layers.Concatenate()([noise_input, label_input])
    x = layers.Dense(5 * 5 * 256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((5, 5, 256))(x)
    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False, name="gen_deconv1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False, name="gen_deconv2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    out = layers.Conv2DTranspose(1, 4, strides=2, padding='same', activation='tanh', use_bias=False)(x)
    return models.Model([noise_input, label_input], out)

def build_discriminator():
    image_input = layers.Input(shape=IMG_SHAPE)
    label_input = layers.Input(shape=(NUM_CLASSES,))
    label_map = layers.Dense(np.prod(IMG_SHAPE))(label_input)
    label_map = layers.Reshape(IMG_SHAPE)(label_map)
    merged = layers.Concatenate()([image_input, label_map])
    x = layers.Conv2D(64, 4, strides=2, padding='same', name="disc_conv1")(merged)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(128, 4, strides=2, padding='same', name="disc_conv2")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    return models.Model([image_input, label_input], out)

# -------------------------
# Save Generator Outputs
# -------------------------
def save_samples(generator, epoch, n=10):
    noise = np.random.normal(0, 1, (n, 100))
    labels = np.arange(n) % 9  # Generate labels for classes 0 to 8
    labels = labels.reshape(-1, 1)

    labels = tf.keras.utils.to_categorical(labels, num_classes=9)
    generated_samples = generator.predict([noise, labels], verbose=0)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
    for i in range(n):
        axes[i].imshow(generated_samples[i].reshape(40, 40), cmap='gray')
        axes[i].set_title(f"Class {labels[i][0]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(f"cdgan_generated_epoch_{epoch}.png")
    plt.close()

# -------------------------------
# Training Loop with TensorBoard Logging
# -------------------------------
def train_cdgan():
    log_message("Initializing Conditional Generator...")
    generator = build_generator()
    log_message("Generator initialized.")

    log_message("Initializing Conditional Discriminator...")
    discriminator = build_discriminator()
    log_message("Discriminator initialized.")

    d_optimizer = Adam(LR, BETA_1)
    g_optimizer = Adam(LR, BETA_1)

    discriminator.compile(optimizer=d_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    noise_input = layers.Input(shape=(LATENT_DIM,))
    label_input = layers.Input(shape=(NUM_CLASSES,))
    gen_img = generator([noise_input, label_input])
    discriminator.trainable = False
    validity = discriminator([gen_img, label_input])
    combined = models.Model([noise_input, label_input], validity)
    combined.compile(optimizer=g_optimizer, loss='binary_crossentropy')

    for epoch in trange(EPOCHS):
        log_message(f"Epoch {epoch + 1}/{EPOCHS} starting...")
        start_time = time.time()

        idx = np.random.randint(0, train_data.shape[0], BATCH_SIZE)
        real_imgs = train_data[idx]
        real_lbls = train_labels[idx]
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        gen_imgs = generator.predict([noise, real_lbls], verbose=0)

        real_y = np.random.uniform(0.85, 1.0, size=(BATCH_SIZE, 1))
        fake_y = np.random.uniform(0.0, 0.15, size=(BATCH_SIZE, 1))

        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch([real_imgs, real_lbls], real_y)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, real_lbls], fake_y)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        discriminator.trainable = False
        misleading_y = np.ones((BATCH_SIZE, 1))
        g_loss = combined.train_on_batch([noise, real_lbls], misleading_y)

        fid = None

        if (epoch + 1) % 1000 == 0:
            # Example logging (use your actual variable names):
            log_message(f"Generator Loss: {g_loss:.4f}")
            log_message(f"Discriminator Loss: {d_loss[0]:.4f}")
            log_message(f"Discriminator Accuracy: {d_loss[1] * 100:.2f}%")

            log_message(f"Discriminator Real Accuracy: {d_loss_real[1] * 100:.2f}%")
            log_message(f"Discriminator Fake Accuracy: {d_loss_fake[1] * 100:.2f}%")

            fid = calculate_fid(real_imgs[:100], gen_imgs[:100])
            log_message(f"FID Score at Epoch {epoch + 1}: {fid:.4f}")

            save_samples(generator, epoch)
            log_message(f"Generated samples saved at epoch {epoch + 1}.")

            # Save weights
            generator.save_weights(f"cdgan_checkpoints/generator_epoch_{epoch}.h5")
            discriminator.save_weights(f"cdgan_checkpoints/discriminator_epoch_{epoch}.h5")
            log_message(f"Model weights saved at epoch {epoch + 1}.")

        # TensorBoard logging
        with summary_writer.as_default():
            tf.summary.scalar("Generator Loss", g_loss, step=epoch)
            tf.summary.scalar("Discriminator Loss", d_loss[0], step=epoch)
            tf.summary.scalar("Discriminator Accuracy", d_loss[1], step=epoch)
            if fid is not None:
                tf.summary.scalar("FID Score", fid, step=epoch)
            tf.summary.scalar("Epoch Time (s)", time.time() - start_time, step=epoch)

    log_message("Training complete.")


if __name__ == "__main__":
    train_cdgan()