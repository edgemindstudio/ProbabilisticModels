import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, LeakyReLU, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow_gan import eval as tfgan_eval
from tensorflow_gan import losses as tfgan_losses
from tqdm import trange
from scipy.linalg import sqrtm
import datetime
import tensorflow_gan as tfgan
import tensorflow_hub as hub
import tensorflow_probability as tfp

# -------------------------
# Generator Model
# -------------------------
def build_generator():
    model = Sequential([
        Input(shape=(100,)),
        Dense(1024), BatchNormalization(), LeakyReLU(alpha=0.2), Dropout(0.2),
        Dense(512), BatchNormalization(), LeakyReLU(alpha=0.2), Dropout(0.2),
        Dense(256), BatchNormalization(), LeakyReLU(alpha=0.2), Dropout(0.2),
        Dense(1600, activation='tanh'), Reshape((40, 40, 1))
    ])
    return model

# -------------------------
# Discriminator Model (Reduced size slightly + dropout reduced)
# -------------------------
def build_discriminator():
    model = Sequential([
        Input(shape=(40, 40, 1)), Flatten(),

        Dense(256),  # ↓ Reduced from 384 to 256 to avoid overpowering G
        LeakyReLU(alpha=0.2),
        Dropout(0.2),  # ↓ Reduced from 0.3 for less aggressive regularization

        Dense(64),  # ↓ Reduced from 128
        LeakyReLU(alpha=0.2),
        Dropout(0.2),  # ↓ Reduced dropout

        Dense(1, activation='sigmoid')
    ])
    return model

# -------------------------
# Load and Prepare Dataset
# -------------------------
#datafile_path = r"C:\Users\fonke\Downloads\USTC-TFC2016_malware"
datafile_path = "./USTC-TFC2016_malware"
train_data = np.load(datafile_path + "/train_data.npy", allow_pickle=True)
train_labels = np.load(datafile_path + "/train_labels.npy", allow_pickle=True)
test_data = np.load(datafile_path + "/test_data.npy", allow_pickle=True)
test_labels = np.load(datafile_path + "/test_labels.npy", allow_pickle=True)

train_data = train_data.reshape(-1, 40, 40, 1)
test_data = test_data.reshape(-1, 40, 40, 1)

# Normalize to [-1, 1] for tanh activation
train_data = (train_data - 0.5) * 2.0
test_data = (test_data - 0.5) * 2.0

# Normalization for Sigmoid Activation
# train_data = train_data / 255.0 # For Sigmoi
# test_data = test_data / 255.0

# -------------------------
# Initialize models
# -------------------------
generator = build_generator()
discriminator = build_discriminator()

# 👇 Must set trainable before compiling!
discriminator.trainable = True
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(0.00002, 0.5),  # ↓ Reduced from 0.00005
    metrics=['accuracy']
)

# GAN MODEL
z = tf.keras.Input(shape=(100,))
generated_image = generator(z)
# 👇 Freeze D before compiling the GAN
discriminator.trainable = False
validity = discriminator(generated_image)
# 👇 Compiling the GAN
gan = tf.keras.Model(z, validity)
gan.compile(
    loss='binary_crossentropy',
    optimizer=Adam(0.0003, 0.5)  # ↑ Increased from 0.0002
)

# -------------------------
# Helper: Generate synthetic samples
# -------------------------
def generate_synthetic_samples(generator, num_samples=10):
    noise = np.random.normal(0, 1, (num_samples, 100))
    generated_samples = generator.predict(noise)
    generated_samples = ((generated_samples + 1e-7) * 255).astype(np.uint8)
    return generated_samples

# -------------------------
# Helper: Evaluate Discriminator
# -------------------------
def evaluate_discriminator(discriminator, real_data, generator, num_samples=100):
    noise = np.random.normal(0, 1, (num_samples, 100))
    fake_data = generator.predict(noise)

    real_labels = np.ones((num_samples, 1)) * 0.85  # ↑ Label smoothing
    fake_labels = np.zeros((num_samples, 1)) + 0.1  # ↑ Fake label smoothing

    real_loss, real_acc = discriminator.evaluate(real_data[:num_samples], real_labels, verbose=0)
    fake_loss, fake_acc = discriminator.evaluate(fake_data, fake_labels, verbose=0)

    print(f"Discriminator Accuracy on Real Data: {real_acc * 100:.2f}%")
    print(f"Discriminator Accuracy on Fake Data: {fake_acc * 100:.2f}%")

    return real_acc, fake_acc

# -------------------------
# Helper: Visualize Generator Output
# -------------------------
def visualize_generated_samples(generator, num_samples=10, epoch=0):
    generated_samples = generate_synthetic_samples(generator, num_samples)
    fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
    for i in range(num_samples):
        axes[i].imshow(generated_samples[i].reshape(40, 40), cmap='gray')
        axes[i].axis('off')
    plt.savefig(f"generated_sample_epoch_{epoch}.png")  # Or use dynamic name
    plt.close() # ✅ Avoid backend crash

# -------------------------
# FID Calculation Function
# -------------------------
# Load InceptionV3 feature extractor from TensorFlow Hub (used in FID)
inception_model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(75, 75, 3))
def calculate_fid(real_images, generated_images):
    # Resize to 75x75x3 (InceptionV3 input requirement)
    real_images_resized = tf.image.resize(real_images, (75, 75))
    fake_images_resized = tf.image.resize(generated_images, (75, 75))

    # Stack to 3 channels
    real_images_resized = tf.repeat(real_images_resized, repeats=3, axis=-1)
    fake_images_resized = tf.repeat(fake_images_resized, repeats=3, axis=-1)

    # Extract features
    real_activations = inception_model(real_images_resized)
    fake_activations = inception_model(fake_images_resized)

    # Calculate mean and covariance
    mu1, sigma1 = tf.reduce_mean(real_activations, axis=0), tfp.stats.covariance(real_activations)
    mu2, sigma2 = tf.reduce_mean(fake_activations, axis=0), tfp.stats.covariance(fake_activations)

    # FID formula
    ssdiff = tf.reduce_sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + tf.linalg.trace(sigma1 + sigma2 - 2 * covmean)
    return fid.numpy()

# -------------------------
# Logger
# -------------------------
def log_results(epoch, d_loss, g_loss, real_acc, fake_acc, log_file="gan_training_log.txt"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (f"[{timestamp}] Epoch {epoch}: D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}, "
                 f"Real Acc: {real_acc:.2f}%, Fake Acc: {fake_acc:.2f}%\n")
    print(log_entry.strip())
    with open(log_file, "a") as f:
        f.write(log_entry)

# -------------------------
# Training Loop
# -------------------------
def train_gan(epochs=10000, batch_size=256):  # ↑ Increased from 128 to 256
    log_file = "gan_training_log.txt"
    d_losses, g_losses = [], []

    for epoch in range(epochs):
        # Sample and slightly perturb real images
        idx = np.random.randint(0, train_data.shape[0], batch_size)
        real_images = train_data[idx] #+ np.random.normal(0, 0.02, train_data[idx].shape)  # ↓ Reduced noise

        # Generate fake images
        noise = np.random.normal(0, 1, (batch_size, 100))
        fake_images = generator.predict(noise)
        fake_images += np.random.normal(0, 0.02, fake_images.shape)  # ↓ Reduced noise

        # Smoothing labels
        real_labels = np.ones((batch_size, 1)) #* 0.85  # ↑ Real label smoothing
        fake_labels = np.zeros((batch_size, 1)) #+ 0.1  # ↑ Fake label smoothing

        # ✅ Set discriminator trainable before training
        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ✅ Freeze discriminator before training generator
        discriminator.trainable = False
        noise = np.random.normal(0, 1, (batch_size, 100))
        misleading_labels = np.ones((batch_size, 1)) * 0.85
        g_loss = gan.train_on_batch(noise, misleading_labels)

        # ✅ Track losses for plotting
        d_losses.append(d_loss[0])
        g_losses.append(g_loss)

        # Log and evaluate
        if epoch % 1000 == 0:
            real_acc, fake_acc = evaluate_discriminator(discriminator, train_data, generator)
            log_results(epoch, d_loss[0], g_loss, real_acc * 100, fake_acc * 100, log_file)
            visualize_generated_samples(generator, num_samples=5, epoch=epoch)

            # FID
            fid = calculate_fid(train_data[:100], generator.predict(np.random.normal(0, 1, (100, 100))))
            print(f"FID Score: {fid:.2f}")

            # Save Model Weights
            generator.save_weights(f"checkpoints/generator_epoch_{epoch}.h5")
            discriminator.save_weights(f"checkpoints/discriminator_epoch_{epoch}.h5")

            # Visualize Generator Output Range
            fake_sample = generator.predict(np.random.normal(0, 1, (1, 100)))
            print("Fake Output Min:", np.min(fake_sample), "Max:", np.max(fake_sample))

    # ✅ Final plotting after all epochs training
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Curves')
    plt.grid(True)
    plt.savefig("training_loss_plot.png")
    #plt.show()
    plt.close()  # ✅ Prevent Matplotlib crash in PyCharm

# -------------------------
# Start Training
# -------------------------
os.makedirs("checkpoints", exist_ok=True)  # ✅ Prevents file I/O errors
train_gan(epochs=10000, batch_size=256)

# -------------------------
# Final Evaluation
# -------------------------
visualize_generated_samples(generator, num_samples=10, epoch="final")
evaluate_discriminator(discriminator, train_data, generator, num_samples=500)
