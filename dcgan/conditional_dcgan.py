import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tqdm import trange
from tensorflow.keras.optimizers.legacy import Adam
from scipy.linalg import sqrtm
import tensorflow_probability as tfp

#-------------------------------
# Config
#-------------------------------
IMG_SHAPE = (40, 40, 1)
LATENT_DIM = 100
NUM_CLASSES = 9
BATCH_SIZE = 256
EPOCHS = 10000
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "USTC-TFC2016_malware")
os.makedirs("cdgan_checkpoints", exist_ok=True)

#-------------------------------
# Load & Normalize Data [-1, 1] for tanh
#-------------------------------
train_data = np.load(os.path.join(DATA_PATH, "train_data.npy"), allow_pickle=True)
train_labels = np.load(os.path.join(DATA_PATH, "train_labels.npy"), allow_pickle=True)
train_data = train_data.reshape(-1, *IMG_SHAPE)
train_data = (train_data - 0.5) * 2.0
train_labels = tf.keras.utils.to_categorical(train_labels, NUM_CLASSES)

# -------------------------
# FID Calculation (Inception)
# -------------------------
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

# -------------------------
# Conditional Generator
# -------------------------
def build_generator():
    noise_input = layers.Input(shape=(LATENT_DIM,))
    label_input = layers.Input(shape=(NUM_CLASSES,))
    x = layers.Concatenate()([noise_input, label_input])
    x = layers.Dense(5 * 5 * 256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((5, 5, 256))(x)
    x = layers.Conv2DTranspose(128, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2DTranspose(64, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    out = layers.Conv2DTranspose(1, 4, strides=2, padding='same', activation='tanh', use_bias=False)(x)
    model = models.Model([noise_input, label_input], out, name="Conditional_Generator")
    model.summary()
    return model

# -------------------------
# Conditional Discriminator
# -------------------------
def build_discriminator():
    image_input = layers.Input(shape=IMG_SHAPE)
    label_input = layers.Input(shape=(NUM_CLASSES,))
    label_map = layers.Dense(np.prod(IMG_SHAPE))(label_input)
    label_map = layers.Reshape(IMG_SHAPE)(label_map)
    merged = layers.Concatenate()([image_input, label_map])
    x = layers.Conv2D(64, 4, strides=2, padding='same')(merged)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model([image_input, label_input], out, name="Conditional_Discriminator")
    model.summary()
    return model

# -------------------------
# Save Generator Outputs
# -------------------------
def save_samples(generator, epoch):
    noise = np.random.normal(0, 1, (NUM_CLASSES, LATENT_DIM))
    labels = np.eye(NUM_CLASSES)
    generated = generator.predict([noise, labels])
    generated = np.clip((generated + 1) * 127.5, 0, 255).astype(np.uint8)
    fig, axs = plt.subplots(1, NUM_CLASSES, figsize=(NUM_CLASSES * 2, 2))
    for i in range(NUM_CLASSES):
        axs[i].imshow(generated[i].reshape(40, 40), cmap='gray')
        axs[i].axis('off')
    plt.savefig(f"generated_sample_epoch_{epoch}.png")
    plt.close()

# -------------------------
# Train Conditional GAN
# -------------------------
def train_cdgan():
    generator = build_generator()
    discriminator = build_discriminator()
    discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])
    noise_input = layers.Input(shape=(LATENT_DIM,))
    label_input = layers.Input(shape=(NUM_CLASSES,))
    gen_img = generator([noise_input, label_input])
    discriminator.trainable = False
    validity = discriminator([gen_img, label_input])
    combined = models.Model([noise_input, label_input], validity)
    combined.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

    for epoch in trange(EPOCHS):
        idx = np.random.randint(0, train_data.shape[0], BATCH_SIZE)
        real_imgs = train_data[idx]
        real_lbls = train_labels[idx]
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        gen_imgs = generator.predict([noise, real_lbls])
        real_y = np.random.uniform(0.85, 1.0, size=(BATCH_SIZE, 1))
        fake_y = np.random.uniform(0.0, 0.15, size=(BATCH_SIZE, 1))
        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch([real_imgs, real_lbls], real_y)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, real_lbls], fake_y)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        discriminator.trainable = False
        misleading_y = np.ones((BATCH_SIZE, 1))
        g_loss = combined.train_on_batch([noise, real_lbls], misleading_y)
        if epoch % 1000 == 0:
            save_samples(generator, epoch)
            print(f"Epoch {epoch} | D Loss: {d_loss[0]:.4f}, G Loss: {g_loss:.4f}, D Acc: {d_loss[1]*100:.2f}%")
            fid = calculate_fid(real_imgs[:100], gen_imgs[:100])
            print(f"FID Score: {fid:.2f}")

if __name__ == "__main__":
    train_cdgan()