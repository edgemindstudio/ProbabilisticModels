# -------------------------------------
# SECTION 1: Imports and Initial Setup
# -------------------------------------
import os
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.linalg import sqrtm
import tensorflow_probability as tfp
from datetime import datetime
import yaml

# Set Random Seeds for Reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Load Configurations from YAML
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# === Conditional CVAE CONFIG ===
EXPERIMENT_NAME = "conditional_cvae"
LOG_FILE = f"{EXPERIMENT_NAME}_result.txt"
CHECKPOINT_DIR = f"{EXPERIMENT_NAME}_checkpoints"
SYNTHETIC_DIR = f"{EXPERIMENT_NAME}_synthetic_samples"
LOG_DIR = os.path.join("logs", EXPERIMENT_NAME)
TENSORBOARD_DIR = os.path.join(LOG_DIR, "tensorboard", datetime.now().strftime("%Y%m%d-%H%M%S"))

# Logging function
VERBOSE = config.get("verbose", True)
def log_message(message, display=True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    if VERBOSE and display:
        print(full_msg)
    with open(LOG_FILE, "a") as f:
        f.write(full_msg + "\n")

# Create necessary directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(SYNTHETIC_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# TensorBoard writer
summary_writer = tf.summary.create_file_writer(TENSORBOARD_DIR)

# Save hyperparameters from config
IMG_SHAPE = tuple(config["IMG_SHAPE"])
LATENT_DIM = config["LATENT_DIM"]
NUM_CLASSES = config["NUM_CLASSES"]
BATCH_SIZE = config["BATCH_SIZE"]
EPOCHS = config["EPOCHS"]
LR = config["LR"]
BETA_1 = config["BETA_1"]
BETA_KL = config.get("BETA_KL", 1.0)

with open(os.path.join(LOG_DIR, "hyperparameters.txt"), "w") as f:
    f.write(f"IMG_SHAPE={IMG_SHAPE}, LATENT_DIM={LATENT_DIM}, NUM_CLASSES={NUM_CLASSES}, "
            f"BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, LR={LR}, BETA_1={BETA_1}\n")

# Dataset location
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "USTC-TFC2016_malware")

NUM_SAMPLES = (NUM_CLASSES * (100 // NUM_CLASSES))  # e.g. 99 if NUM_CLASSES=9

# Fixed noise and labels for consistent evaluation
FIXED_NOISE = np.random.normal(0, 1, (NUM_SAMPLES, LATENT_DIM))
FIXED_LABELS = np.tile(np.arange(NUM_CLASSES), NUM_SAMPLES // NUM_CLASSES).reshape(-1, 1)
FIXED_LABELS_ONEHOT = tf.keras.utils.to_categorical(FIXED_LABELS, NUM_CLASSES)


# Mode selector
RUN_MODE = config.get("mode", "train")  # 'train' or 'eval_only'

# EXPERIMENT 3
# -------------------------------------
# SECTION 2A: Build the Encoder Network
# -------------------------------------
def build_encoder(img_shape, latent_dim, num_classes):
    img_input = layers.Input(shape=img_shape, name="image_input")
    label_input = layers.Input(shape=(num_classes,), name="label_input")

    # Expand and tile label to match image spatial dimensions
    label_expand = layers.Reshape((1, 1, num_classes))(label_input)
    label_tiled = tf.tile(label_expand, [1, img_shape[0], img_shape[1], 1])

    # Concatenate image and label
    x = layers.Concatenate()([img_input, label_tiled])

    # Convolutional layers
    x = layers.Conv2D(32, kernel_size=3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, kernel_size=3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)

    # Latent mean and log variance
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)  # 🔥 Prevent exploding values

    z = Sampling()([z_mean, z_log_var])

    encoder = models.Model([img_input, label_input], [z_mean, z_log_var, z], name="Conditional_Encoder")

    return encoder


# EXPERIMENT 3
# -------------------------------------
# SECTION 2B: Define the Sampling Layer (Reparameterization Trick)
# -------------------------------------
class Sampling(layers.Layer):
    """Samples z ~ N(mu, sigma^2) using reparameterization trick."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# -------------------------------------
# SECTION 2C: Build the Decoder Network
# -------------------------------------
def build_decoder(latent_dim, img_shape, num_classes):
    z_input = layers.Input(shape=(latent_dim,), name="z_input")
    label_input = layers.Input(shape=(num_classes,), name="label_input")

    # Concatenate latent vector and label
    x = layers.Concatenate()([z_input, label_input])

    # Project and reshape into feature map
    x = layers.Dense(10 * 10 * 64, activation="relu")(x)
    x = layers.Reshape((10, 10, 64))(x)

    # Upsample using transposed convolutions
    x = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(1, kernel_size=3, padding="same", activation="tanh")(x)  # output in [-1, 1]

    decoder = models.Model([z_input, label_input], x, name="Decoder")
    return decoder


# -------------------------------------
# SECTION 2D: Assemble the Conditional VAE Model Class
# -------------------------------------
class CVAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.sampling = Sampling()

    def call(self, inputs, training=False):
        img, label = inputs
        z_mean, z_log_var = self.encoder([img, label])
        z = self.sampling([z_mean, z_log_var])
        reconstructed = self.decoder([z, label])
        return reconstructed, z_mean, z_log_var


# -------------------------------------
# SECTION 3: Loss Function and Optimizer
# -------------------------------------

# Reconstruction loss: Binary Crossentropy over pixels (assuming normalized to [-1, 1])
reconstruction_loss_fn = tf.keras.losses.MeanSquaredError()

# KL Divergence: KL(N(μ, σ) || N(0, 1)) = -0.5 * sum(1 + logσ² - μ² - σ²)
def kl_divergence(z_mean, z_log_var):
    return -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)

# EXPERIMENT 3
# Combined loss function
def compute_cvae_loss(x, y, model, beta=1.0):
    z_mean, z_log_var, z = model.encoder([x, y])
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    z = z_mean + tf.exp(0.5 * z_log_var) * epsilon

    reconstruction = model.decoder([z, y])

    reconstruction_loss = tf.reduce_mean(tf.square(x - reconstruction))

    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
    kl_loss = tf.reduce_mean(kl_loss)

    total_loss = reconstruction_loss + beta * kl_loss
    return total_loss, reconstruction_loss, kl_loss


# Optimizer
optimizer = Adam(learning_rate=LR, beta_1=BETA_1)


# -------------------------------------
# SECTION 4: Updated Training Loop (with Early Stopping & Logging)
# -------------------------------------
@tf.function
def train_step(model, x_batch, y_batch):
    """Single training step for one batch."""
    with tf.GradientTape() as tape:
        total_loss, rec_loss, kl_loss = compute_cvae_loss(model, x_batch, y_batch)
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss, rec_loss, kl_loss


def validate(model, val_dataset):
    """Evaluate loss on the validation dataset."""
    val_losses = []
    for x_val, y_val in val_dataset:
        val_loss, _, _ = compute_cvae_loss(model, x_val, y_val)
        val_losses.append(val_loss.numpy())
    return np.mean(val_losses)

# EXPERIMENT 3
def train_cvae(model, train_dataset, val_dataset, epochs, summary_writer):
    best_val_loss = float('inf')
    patience_counter = 0

    log_message("Starting CVAE training with early stopping...")
    beta_history = []  # Add this at the beginning of training loop

    for epoch in trange(epochs, desc="Epochs", position=0):
        start_time = time.time()

        # === β Warm-up: Gradually increase from 0 to 1 over 50 epochs
        beta = min(1.0, epoch / 50.0)
        beta_history.append(beta)  # Track it per epoch

        train_losses, recon_losses, kl_losses = [], [], []

        for x_batch, y_batch in train_dataset:
            with tf.GradientTape() as tape:
                total_loss, recon_loss, kl_loss = compute_cvae_loss(x_batch, y_batch, model, beta)

            grads = tape.gradient(total_loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

            train_losses.append(total_loss.numpy())
            recon_losses.append(recon_loss.numpy())
            kl_losses.append(kl_loss.numpy())

        # === Validation
        val_losses = []
        for x_batch, y_batch in val_dataset:
            val_loss, _, _ = compute_cvae_loss(x_batch, y_batch, model, beta)
            val_losses.append(val_loss.numpy())

        avg_train_loss = np.mean(train_losses)
        avg_recon_loss = np.mean(recon_losses)
        avg_kl_loss = np.mean(kl_losses)
        avg_val_loss = np.mean(val_losses)

        # === Logging
        with summary_writer.as_default():
            tf.summary.scalar("Train/Total_Loss", avg_train_loss, step=epoch)
            tf.summary.scalar("Train/Reconstruction_Loss", avg_recon_loss, step=epoch)
            tf.summary.scalar("Train/KL_Loss", avg_kl_loss, step=epoch)
            tf.summary.scalar("Val/Total_Loss", avg_val_loss, step=epoch)
            tf.summary.scalar("Beta", beta, step=epoch)

        log_message(
            f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Recon = {avg_recon_loss:.4f}, KL = {avg_kl_loss:.4f}, Val Loss = {avg_val_loss:.4f}, β = {beta:.2f}"
        )

        # === Early stopping and model checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_cvae_weights(model.encoder, model.decoder, "best")
            patience_counter = 0
            log_message("New best model saved.")
        else:
            patience_counter += 1
            log_message(f"Patience Counter: {patience_counter}/10")
            if patience_counter >= 10:
                log_message("Early stopping triggered.")
                break

    log_message("Training complete.")
    plot_beta_schedule(beta_history)


# EXPERIMENT 3
# --------------------------------------
# SECTION 4B: Plot β warm-up curve
# --------------------------------------
def plot_beta_schedule(beta_history):
    plt.figure(figsize=(8, 4))
    plt.plot(range(len(beta_history)), beta_history, marker='o')
    plt.title("β-VAE Warm-up Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("β Value")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "beta_schedule.png"))
    plt.close()


# -------------------------------------
# SECTION 5: Data Loading and Preprocessing
# -------------------------------------
def load_malware_dataset(data_path, img_shape, num_classes):
    x_train = np.load(os.path.join(data_path, "train_data.npy"))
    y_train = np.load(os.path.join(data_path, "train_labels.npy"))
    x_test = np.load(os.path.join(data_path, "test_data.npy"))
    y_test = np.load(os.path.join(data_path, "test_labels.npy"))

    # Normalize to [-1, 1]
    x_train = (x_train.astype("float32") - 127.5) / 127.5
    x_test = (x_test.astype("float32") - 127.5) / 127.5

    # Ensure correct shape
    x_train = np.reshape(x_train, (-1, *img_shape))
    x_test = np.reshape(x_test, (-1, *img_shape))

    # One-hot encode labels
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)

def create_datasets(batch_size):
    (x_train, y_train), (x_test, y_test) = load_malware_dataset(DATA_PATH, IMG_SHAPE, NUM_CLASSES)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=10240).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset


# -------------------------------------
# SECTION 6A: Generate and Visualize Conditional Samples
# -------------------------------------
def generate_and_plot_samples(model, fixed_noise, fixed_labels, num_classes, save_path=None):
    log_message("Generating synthetic samples for visualization...")
    generated_images = model.decoder([fixed_noise, fixed_labels], training=False)

    generated_images = (generated_images + 1.0) / 2.0  # scale from [-1, 1] to [0, 1]
    generated_images = generated_images.numpy()

    plt.figure(figsize=(12, 2))
    for i in range(num_classes):
        plt.subplot(1, num_classes, i + 1)
        plt.imshow(generated_images[i].reshape(IMG_SHAPE[0], IMG_SHAPE[1]), cmap="gray")
        plt.axis("off")
        plt.title(f"Class {i}")
    if save_path:
        plt.savefig(save_path)
    plt.close()


# -------------------------------------
# SECTION 6B: Compute FID Score Between Real and Generated Samples
# -------------------------------------
def calculate_fid(real_images, generated_images):
    """Computes FID score between real and generated samples using Inception features."""
    real_images = tf.image.resize(real_images, (299, 299))
    generated_images = tf.image.resize(generated_images, (299, 299))

    # Convert grayscale to RGB by duplicating channels
    real_images_rgb = tf.image.grayscale_to_rgb(real_images)
    generated_images_rgb = tf.image.grayscale_to_rgb(generated_images)

    # Preprocess for InceptionV3
    real_images_rgb = tf.keras.applications.inception_v3.preprocess_input(real_images_rgb)
    generated_images_rgb = tf.keras.applications.inception_v3.preprocess_input(generated_images_rgb)

    # Load InceptionV3 model
    inception = tf.keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    # Extract features
    act1 = inception.predict(real_images_rgb, verbose=0)
    act2 = inception.predict(generated_images_rgb, verbose=0)

    # Calculate mean and covariance
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    # Compute FID (Fréchet Distance)
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1.dot(sigma2))

    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


# -------------------------------------
# SECTION 6C: Classifier Accuracy on Synthetic Samples
# -------------------------------------
def evaluate_classifier_accuracy(classifier_model, decoder, latent_dim, num_classes, n_samples=1000):
    log_message("Evaluating synthetic sample classification accuracy...")

    # Generate synthetic data
    z = np.random.normal(0, 1, (n_samples, latent_dim))
    labels = np.random.randint(0, num_classes, size=n_samples).reshape(-1, 1)
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes)

    generated_images = decoder([z, one_hot_labels], training=False)
    generated_images = (generated_images + 1.0) / 2.0  # Rescale to [0, 1]

    # Predict using the pre-trained classifier
    predictions = classifier_model.predict(generated_images, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = labels.flatten()

    accuracy = np.mean(predicted_labels == true_labels)
    log_message(f"Classifier Accuracy on CVAE synthetic samples: {accuracy:.4f}")
    return accuracy


# -------------------------------------
# SECTION 7: Latent Interpolation Between Two Labels
# -------------------------------------
def interpolate_latent_space(decoder, label_start, label_end, latent_dim, steps=10, save_dir="interpolations"):
    """Interpolates in latent space between two random z vectors for two classes."""
    z1 = np.random.normal(0, 1, (1, latent_dim))
    z2 = np.random.normal(0, 1, (1, latent_dim))

    # One-hot labels
    y1 = tf.keras.utils.to_categorical([label_start], NUM_CLASSES)
    y2 = tf.keras.utils.to_categorical([label_end], NUM_CLASSES)

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(steps, 2))
    for i, alpha in enumerate(np.linspace(0, 1, steps)):
        z = (1 - alpha) * z1 + alpha * z2
        y = (1 - alpha) * y1 + alpha * y2
        img = decoder([z, y], training=False)
        img = (img + 1.0) / 2.0  # scale to [0, 1]

        plt.subplot(1, steps, i + 1)
        plt.imshow(img[0].numpy().reshape(IMG_SHAPE[0], IMG_SHAPE[1]), cmap="gray")
        plt.axis("off")
        plt.title(f"{alpha:.1f}")

    plt.suptitle(f"Interpolation: Class {label_start} → {label_end}")

    # Save figure
    filename = f"interp_{label_start}_to_{label_end}.png"
    save_path = os.path.join(save_dir, filename)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    log_message(f"Saved interpolation to {save_path}")

    plt.close()


# EXPERIMENT 2
# -------------------------------------
# SECTION 7B: Latent Interpolation Function
# -------------------------------------
def interpolate_between_classes(encoder, decoder, class_a, class_b, train_dataset, num_steps=10, latent_dim=LATENT_DIM):
    """
    Interpolates between the latent vectors of two classes and visualizes the transition.
    """

    # Fetch a batch from the dataset
    for x_batch, y_batch in train_dataset.take(1):
        x_batch = x_batch.numpy()
        y_batch = np.argmax(y_batch.numpy(), axis=1)

        # Get a sample image from each class
        sample_image_a = x_batch[y_batch == class_a][0:1]
        sample_image_b = x_batch[y_batch == class_b][0:1]
        break

        # Encode both images
        z_mean_a, _, _ = encoder([sample_image_a, tf.one_hot([class_a], NUM_CLASSES)], training=False)
        z_mean_b, _, _ = encoder([sample_image_b, tf.one_hot([class_b], NUM_CLASSES)], training=False)

        # Interpolate between them
        alphas = np.linspace(0, 1, num_steps)
        interpolated = [(1 - a) * z_mean_a + a * z_mean_b for a in alphas]
        interpolated = tf.concat(interpolated, axis=0)

        labels = tf.one_hot([class_a] * num_steps, NUM_CLASSES)
        generated = decoder([interpolated, labels], training=False).numpy()

        # Plot
        plt.figure(figsize=(20, 2))
        for i in range(num_steps):
            plt.subplot(1, num_steps, i + 1)
            plt.imshow(generated[i].squeeze(), cmap="gray")
            plt.axis("off")
        plt.suptitle(f"Interpolation: Class {class_a} → {class_b}")

        # Save the figure
        save_path = f"{EXPERIMENT_NAME}_interpolation_{class_a}_to_{class_b}.png"
        plt.savefig(save_path)
        log_message(f"Saved latent interpolation to {save_path}")
        plt.close()


# -------------------------------------
# SECTION 8: Model Save/Load Utilities
# -------------------------------------
def save_cvae_weights(encoder, decoder, step_name="final"):
    encoder.save_weights(os.path.join(CHECKPOINT_DIR, f"encoder_{step_name}.h5"))
    decoder.save_weights(os.path.join(CHECKPOINT_DIR, f"decoder_{step_name}.h5"))

def load_cvae_weights(encoder, decoder, step_name="best"):
    encoder_path = os.path.join(CHECKPOINT_DIR, f"encoder_{step_name}.h5")
    decoder_path = os.path.join(CHECKPOINT_DIR, f"decoder_{step_name}.h5")
    encoder.load_weights(encoder_path)
    decoder.load_weights(decoder_path)
    log_message(f"Loaded encoder from {encoder_path} and decoder from {decoder_path}")


# -------------------------------------
# SECTION 9: CNN Classifier for Evaluation
# -------------------------------------
def build_classifier(input_shape=(40, 40, 1), num_classes=9):
    model = models.Sequential(name="Improved_Classifier")

    # Block 1
    model.add(layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())

    # Block 2
    model.add(layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())

    # Block 3
    model.add(layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D())

    # Regularization
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))

    # Dense Layers
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))

    # Output
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# -------------------------------------
# SECTION 9B: CNN Classifier Loading or Training
# -------------------------------------
def load_or_train_classifier(train_data, val_data, classifier_ckpt_path="classifier_model.h5"):
    if os.path.exists(classifier_ckpt_path):
        try:
            log_message(f"Loading pre-trained classifier from {classifier_ckpt_path}")
            classifier = build_classifier(input_shape=IMG_SHAPE, num_classes=NUM_CLASSES)
            classifier.load_weights(classifier_ckpt_path)
            return classifier
        except ValueError as e:
            log_message("Layer mismatch error detected. Rebuilding classifier...", display=True)
            os.remove(classifier_ckpt_path)

    log_message("Training a real-only CNN classifier...")
    classifier = build_classifier(input_shape=IMG_SHAPE, num_classes=NUM_CLASSES)

    history = classifier.fit(
        train_data,
        validation_data=val_data,
        epochs=15,
        verbose=1
    )

    classifier.save_weights(classifier_ckpt_path)
    log_message("Classifier trained and weights saved.")
    return classifier


# -------------------------------------
# SECTION 10: Train & Evaluate Real-Only Baseline Classifier
# -------------------------------------
def train_and_evaluate_real_classifier():
    log_message("Running baseline: Real-only CNN classifier...")

    (x_train, y_train), (x_test, y_test) = load_malware_dataset(DATA_PATH, IMG_SHAPE, NUM_CLASSES)

    # Prepare datasets
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    classifier = load_or_train_classifier(train_data, test_data)

    # Evaluate
    test_loss, test_acc = classifier.evaluate(test_data, verbose=0)
    log_message(f"Real-Only Classifier Accuracy on Test Set: {test_acc:.4f}")

    return classifier, test_acc


# -------------------------------------
# SECTION 11: Load CVAE Synthetic Samples
# -------------------------------------
def load_synthetic_samples(synthetic_dir=SYNTHETIC_DIR):
    images_path = os.path.join(synthetic_dir, "images.npy")
    labels_path = os.path.join(synthetic_dir, "labels.npy")

    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        log_message("Synthetic samples not found in expected location.", display=True)
        return None, None

    log_message(f"Loading synthetic samples from {synthetic_dir}")
    x_synth = np.load(images_path)
    y_synth = np.load(labels_path)

    # Convert to TensorFlow dataset
    synth_dataset = tf.data.Dataset.from_tensor_slices((x_synth, y_synth))
    synth_dataset = synth_dataset.shuffle(2048).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return synth_dataset


# -------------------------------------
# SECTION 12: Save Generator Outputs for Visual Preview
# -------------------------------------
def save_visual_preview(model, fixed_noise, fixed_labels, num_classes, save_path=None):
    log_message("Saving visual preview of synthetic samples...")

    generated_images = model.decoder([fixed_noise, fixed_labels], training=False)
    generated_images = (generated_images + 1.0) / 2.0  # [-1, 1] → [0, 1]
    generated_images = generated_images.numpy()

    plt.figure(figsize=(num_classes * 1.5, 2))
    for i in range(num_classes):
        plt.subplot(1, num_classes, i + 1)
        plt.imshow(generated_images[i].reshape(IMG_SHAPE[0], IMG_SHAPE[1]), cmap="gray")
        plt.axis("off")
        plt.title(f"Class {i}")
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        log_message(f"Saved visual sample preview to {save_path}")
    plt.close()


# -------------------------------------
# SECTION 13: Preview Synthetic Samples from Disk
# -------------------------------------
def preview_saved_synthetic_samples(n_show=9, save_path="synthetic_preview.png"):
    log_message("Previewing saved synthetic samples from disk...")

    img_path = os.path.join(SYNTHETIC_DIR, "images.npy")
    label_path = os.path.join(SYNTHETIC_DIR, "labels.npy")

    if not os.path.exists(img_path) or not os.path.exists(label_path):
        log_message("Synthetic sample files not found.", display=True)
        return

    x_synth = np.load(img_path)
    y_synth = np.load(label_path)
    y_labels = np.argmax(y_synth, axis=1)

    plt.figure(figsize=(12, 2))
    for i in range(min(n_show, x_synth.shape[0])):
        plt.subplot(1, n_show, i + 1)
        plt.imshow(x_synth[i].reshape(IMG_SHAPE[0], IMG_SHAPE[1]), cmap="gray")
        plt.axis("off")
        plt.title(f"Class {y_labels[i]}")

    plt.tight_layout()

    # Save the figure
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    log_message(f"Saved preview to {save_path}")

    plt.close()


# -------------------------------------
# SECTION 14: Generate Synthetic Samples Per Class
# -------------------------------------
def generate_synthetic_per_class(model, latent_dim, num_classes, samples_per_class=1000):
    log_message(f"Generating {samples_per_class} synthetic samples per class...")

    all_images = []
    all_labels = []

    for cls in range(num_classes):
        z = np.random.normal(0, 1, (samples_per_class, latent_dim))
        y = tf.keras.utils.to_categorical([cls] * samples_per_class, num_classes)
        images = model.decoder([z, y], training=False)
        images = (images + 1.0) / 2.0  # Scale to [0, 1]

        all_images.append(images.numpy())
        all_labels.append(y)

    x_synth = np.concatenate(all_images, axis=0)
    y_synth = np.concatenate(all_labels, axis=0)

    # Save to disk
    np.save(os.path.join(SYNTHETIC_DIR, "images_per_class.npy"), x_synth)
    np.save(os.path.join(SYNTHETIC_DIR, "labels_per_class.npy"), y_synth)

    log_message(f"Saved class-balanced synthetic dataset: {x_synth.shape}")


# -------------------------------------
# SECTION 15: Load Synthetic Samples (Per-Class)
# -------------------------------------
def load_synthetic_samples():
    log_message("Loading synthetic samples from disk...")

    img_path = os.path.join(SYNTHETIC_DIR, "images_per_class.npy")
    label_path = os.path.join(SYNTHETIC_DIR, "labels_per_class.npy")

    if not os.path.exists(img_path) or not os.path.exists(label_path):
        log_message("Synthetic sample files not found.", display=True)
        return None, None

    x_synth = np.load(img_path)
    y_synth = np.load(label_path)

    log_message(f"Loaded synthetic dataset: {x_synth.shape[0]} samples.")
    return x_synth, y_synth


# -------------------------------------
# SECTION 16: Evaluate Classifier on Real vs Real+Synthetic
# -------------------------------------
def evaluate_real_vs_synthetic(x_train_real, y_train_real, x_test_real, y_test_real,
                                x_synth=None, y_synth=None, use_synthetic=False):
    log_message("Evaluating classifier on real vs real+synthetic training data...")

    if use_synthetic and x_synth is not None:
        x_train = np.concatenate([x_train_real, x_synth])
        y_train = np.concatenate([y_train_real, y_synth])
        log_message(f"Using combined training data: real ({x_train_real.shape[0]}) + synthetic ({x_synth.shape[0]})")
    else:
        x_train = x_train_real
        y_train = y_train_real
        log_message(f"Using real-only training data: {x_train.shape[0]} samples")

    # Build and train classifier
    classifier = build_classifier(input_shape=IMG_SHAPE, num_classes=NUM_CLASSES)
    classifier.fit(
        x_train, y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    # Evaluate on test set
    test_loss, test_accuracy = classifier.evaluate(x_test_real, y_test_real, verbose=0)
    log_message(f"Classifier Test Accuracy: {test_accuracy:.4f}")

    # Optionally save for reuse
    save_classifier_model(classifier)

    return test_accuracy


# -------------------------------------
# SECTION 16B: Save Classifier Model Utility
# -------------------------------------
def save_classifier_model(model, save_path="classifier_real_plus_synthetic.h5"):
    """
    Saves the trained classifier model to the specified path.

    Args:
        model (tf.keras.Model): The trained classifier model.
        save_path (str): Path to save the model file.
    """
    model.save(save_path)
    log_message(f"Saved classifier model to {save_path}")


# -------------------------------------
# SECTION 17: Log Comparison Results
# -------------------------------------
def log_comparison_results(real_acc, combined_acc, output_path=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_path = output_path or os.path.join(LOG_DIR, "comparison_results.txt")

    result = (
        f"\n[{timestamp}] Classifier Performance Comparison\n"
        f"----------------------------------------------\n"
        f"Real-only Training Accuracy     : {real_acc:.4f}\n"
        f"Real + Synthetic Training Acc.  : {combined_acc:.4f}\n"
        f"Difference                      : {combined_acc - real_acc:.4f}\n"
    )

    print(result)
    with open(output_path, "a") as f:
        f.write(result)


# -------------------------------------
# SECTION 18: Main Runner Script (Extended)
# -------------------------------------
if __name__ == "__main__":
    log_message("Initializing Conditional CVAE...")

    # Build model
    encoder = build_encoder(IMG_SHAPE, LATENT_DIM, NUM_CLASSES)
    decoder = build_decoder(LATENT_DIM, IMG_SHAPE, NUM_CLASSES)
    cvae = CVAE(encoder, decoder)
    cvae.optimizer = Adam(learning_rate=1e-3)  # EXPERIMENT 3

    if RUN_MODE == "train":
        # Load data
        train_dataset, val_dataset = create_datasets(BATCH_SIZE)

        # Train model
        train_cvae(cvae, train_dataset, val_dataset, EPOCHS, summary_writer)
        save_cvae_weights(encoder, decoder, "final")

        # EXPERIMENT 2: Visualize interpolations
        interpolate_between_classes(encoder, decoder, class_a=0, class_b=8, train_dataset=train_dataset)

        # Generate + Save visual preview
        generate_and_plot_samples(cvae, FIXED_NOISE, FIXED_LABELS_ONEHOT, NUM_CLASSES)
        save_visual_preview(cvae, FIXED_NOISE, FIXED_LABELS_ONEHOT, NUM_CLASSES)

        # Generate full synthetic dataset
        generate_synthetic_per_class(cvae, LATENT_DIM, NUM_CLASSES)

        # Train & Evaluate classifier
        classifier, real_acc = train_and_evaluate_real_classifier()
        x_synth, y_synth = load_synthetic_samples()
        (x_train, y_train), (x_test, y_test) = load_malware_dataset(DATA_PATH, IMG_SHAPE, NUM_CLASSES)

        combined_acc = evaluate_real_vs_synthetic(x_train, y_train, x_test, y_test, x_synth, y_synth, use_synthetic=True)
        log_comparison_results(real_acc, combined_acc)

        # FID Score
        fid_score = calculate_fid(x_test[:1000], cvae.decoder([FIXED_NOISE, FIXED_LABELS_ONEHOT], training=False))
        log_message(f"FID Score: {fid_score:.4f}")

        # Interpolation
        interpolate_latent_space(cvae.decoder, label_start=0, label_end=8, latent_dim=LATENT_DIM)

    elif RUN_MODE == "eval_only":
        # Load best model
        load_cvae_weights(encoder, decoder, "best")
        cvae = CVAE(encoder, decoder)

        # Generate visuals
        generate_and_plot_samples(cvae, FIXED_NOISE, FIXED_LABELS_ONEHOT, NUM_CLASSES)
        preview_saved_synthetic_samples()

        # Evaluate synthetic accuracy
        classifier, _ = train_and_evaluate_real_classifier()
        acc = evaluate_classifier_accuracy(classifier, cvae.decoder, LATENT_DIM, NUM_CLASSES)
        log_message(f"Classifier Accuracy on Synthetic: {acc:.4f}")