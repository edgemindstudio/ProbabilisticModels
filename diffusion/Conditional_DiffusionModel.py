# =====================================
# SECTION 1: Imports and Initial Setup
# =====================================

import os
import time
import random
from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tqdm import trange
import yaml
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from scipy.linalg import sqrtm
from tensorflow.keras.callbacks import EarlyStopping

# ==============================
# Set Random Seeds
# ==============================
def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_random_seeds()

# ==============================
# Load Configuration File
# ==============================
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# ==============================
# Load Hyperparameters
# ==============================
IMG_SHAPE = tuple(config["IMG_SHAPE"])
NUM_CLASSES = config["NUM_CLASSES"]
BATCH_SIZE = config["BATCH_SIZE"]
EPOCHS = config["EPOCHS"]
LR = config["LR"]
BETA_1 = config["BETA_1"]
RUN_MODE = config.get("mode", "train")  # 'train' or 'eval_only'
VERBOSE = config.get("verbose", True)

# ==============================
# Directory Paths
# ==============================
EXPERIMENT_NAME = "conditional_autoregressive"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "USTC-TFC2016_malware")

LOG_FILE = os.path.join(BASE_DIR, f"{EXPERIMENT_NAME}_result.txt")
CHECKPOINT_DIR = os.path.join(BASE_DIR, f"{EXPERIMENT_NAME}_checkpoints")
SYNTHETIC_DIR = os.path.join(BASE_DIR, f"{EXPERIMENT_NAME}_synthetic_samples")
LOG_DIR = os.path.join(BASE_DIR, "logs", EXPERIMENT_NAME)
TENSORBOARD_DIR = os.path.join(LOG_DIR, "tensorboard", datetime.now().strftime("%Y%m%d-%H%M%S"))

# ==============================
# Logging Utility
# ==============================
def log_message(message, display=True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    if VERBOSE and display:
        print(full_msg)
    with open(LOG_FILE, "a") as f:
        f.write(full_msg + "\n")

# ==============================
# Prepare Output Directories
# ==============================
for dir_path in [CHECKPOINT_DIR, SYNTHETIC_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ==============================
# TensorBoard Summary Writer
# ==============================
summary_writer = tf.summary.create_file_writer(TENSORBOARD_DIR)

# ==============================
# Fixed Labels for Evaluation
# ==============================
FIXED_LABELS = np.arange(NUM_CLASSES).reshape(-1, 1)
FIXED_LABELS_ONEHOT = tf.keras.utils.to_categorical(FIXED_LABELS, NUM_CLASSES)


# =====================================
# SECTION 2: Data Loading and Preprocessing
# =====================================

def load_malware_dataset(data_path, img_shape, num_classes):
    """
    Load and preprocess the malware image dataset.
    Normalizes to [-1, 1], reshapes to (H, W, C), and one-hot encodes labels.
    """
    x_train = np.load(os.path.join(data_path, "train_data.npy"))
    y_train = np.load(os.path.join(data_path, "train_labels.npy"))
    x_test = np.load(os.path.join(data_path, "test_data.npy"))
    y_test = np.load(os.path.join(data_path, "test_labels.npy"))

    # Normalize to [-1, 1]
    # x_train = (x_train.astype(np.float32) - 127.5) / 127.5 # [-1, 1]
    # x_test = (x_test.astype(np.float32) - 127.5) / 127.5 # [-1, 1]

    x_train = x_train.astype(np.float32) / 255.0  # [0, 1]
    x_test = x_test.astype(np.float32) / 255.0  # [0, 1]

    # Reshape to (H, W, C)
    x_train = np.reshape(x_train, (-1, *img_shape))
    x_test = np.reshape(x_test, (-1, *img_shape))

    # One-hot encode labels
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)

def create_datasets(batch_size):
    """
    Create TensorFlow datasets for training, validation, and testing.
    """
    (x_train, y_train), (x_test, y_test) = load_malware_dataset(DATA_PATH, IMG_SHAPE, NUM_CLASSES)

    # Split test set into validation and test sets
    val_split = len(x_test) // 2
    val_data = (x_test[:val_split], y_test[:val_split])
    test_data = (x_test[val_split:], y_test[val_split:])

    # Create and optimize TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=10240).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices(val_data)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset


# -------------------------------------
# SECTION 2A: Noise Scheduler and Beta Schedule
# -------------------------------------
class DiffusionScheduler:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)

        self.alphas = 1.0 - self.betas
        self.alpha_hat = np.cumprod(self.alphas, axis=0)

        self.sqrt_alpha_hat = np.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = np.sqrt(1.0 - self.alpha_hat)

        self.beta_start = beta_start
        self.beta_end = beta_end

    def q_sample(self, x_start, t, noise=None):
        """
        Forward process: add noise to x_start at time t.
        """
        if noise is None:
            noise = tf.random.normal(shape=tf.shape(x_start))

        sqrt_alpha_hat_t = tf.gather(self.sqrt_alpha_hat, t)
        sqrt_one_minus_alpha_hat_t = tf.gather(self.sqrt_one_minus_alpha_hat, t)

        # Expand dims to allow broadcasting
        sqrt_alpha_hat_t = tf.reshape(sqrt_alpha_hat_t, (-1, 1, 1, 1))
        sqrt_one_minus_alpha_hat_t = tf.reshape(sqrt_one_minus_alpha_hat_t, (-1, 1, 1, 1))

        return sqrt_alpha_hat_t * x_start + sqrt_one_minus_alpha_hat_t * noise


# -------------------------------------
# SECTION 2B: Define the Conditional UNet Model (Noise Predictor)
# -------------------------------------
def get_timestep_embedding(timesteps, embedding_dim=128):
    """
    Sinusoidal positional encoding for timestep embedding.
    """
    half_dim = embedding_dim // 2
    exponent = -np.log(10000) / (half_dim - 1)
    emb = tf.cast(timesteps, tf.float32)[:, None] * tf.exp(tf.range(half_dim, dtype=tf.float32) * exponent)
    emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
    return emb

def build_conditional_unet(img_shape, num_classes, embedding_dim=128):
    """
    A small conditional UNet to predict noise given noisy image x_t, timestep t, and label y.
    """
    image_input = layers.Input(shape=img_shape, name="noisy_input")
    timestep_input = layers.Input(shape=(), dtype=tf.int32, name="timestep_input")
    label_input = layers.Input(shape=(num_classes,), name="label_input")

    # --- Embedding for timestep and label ---
    t_embedding = get_timestep_embedding(timestep_input, embedding_dim)
    t_dense = layers.Dense(np.prod(img_shape), activation="relu")(t_embedding)
    t_reshaped = layers.Reshape(img_shape)(t_dense)

    y_dense = layers.Dense(np.prod(img_shape), activation="relu")(label_input)
    y_reshaped = layers.Reshape(img_shape)(y_dense)

    # --- Combine all conditioning ---
    x = layers.Concatenate()([image_input, t_reshaped, y_reshaped])

    # --- Simple UNet-like down-up model ---
    x = layers.Conv2D(64, 3, activation='relu', padding="same")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu', padding="same")(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding="same")(x)
    x = layers.Conv2D(img_shape[-1], 1, activation=None)(x)  # Predict noise

    return models.Model([image_input, timestep_input, label_input], x, name="Conditional_UNet")


# -------------------------------------
# SECTION 3: Loss Function and Optimizer
# -------------------------------------
# MSE loss between predicted noise and actual noise added
mse_loss_fn = tf.keras.losses.MeanSquaredError()

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=BETA_1)


# -------------------------------------
# SECTION 4: Noise Schedule and Diffusion Utilities
# -------------------------------------
def get_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """
    Linear noise schedule from beta_start to beta_end over T steps.
    """
    return np.linspace(beta_start, beta_end, T, dtype=np.float32)

# Set diffusion steps and beta schedule
T = config.get("T", 1000)
betas = get_beta_schedule(T)
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas)

# Precomputed tensors
betas_tf = tf.constant(betas, dtype=tf.float32)
alphas_cumprod_tf = tf.constant(alphas_cumprod, dtype=tf.float32)
sqrt_alphas_cumprod = tf.sqrt(alphas_cumprod_tf)
sqrt_one_minus_alphas_cumprod = tf.sqrt(1.0 - alphas_cumprod_tf)

def add_noise(x_start, noise, t):
    """
    Forward process: q(x_t | x_0)
    Adds noise to the original image at time step t.
    """
    batch_size = tf.shape(x_start)[0]
    sqrt_alpha_cumprod_t = tf.gather(sqrt_alphas_cumprod, t)
    sqrt_one_minus_alpha_cumprod_t = tf.gather(sqrt_one_minus_alphas_cumprod, t)

    sqrt_alpha_cumprod_t = tf.reshape(sqrt_alpha_cumprod_t, [batch_size, 1, 1, 1])
    sqrt_one_minus_alpha_cumprod_t = tf.reshape(sqrt_one_minus_alpha_cumprod_t, [batch_size, 1, 1, 1])

    return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise


# -------------------------------------
# SECTION 5: Define the Denoising Network (U-Net Inspired)
# -------------------------------------
def build_denoising_model(img_shape, num_classes, time_embedding_dim=128):
    """
    Simple U-Net-inspired CNN with time and class conditioning.
    """
    img_input = layers.Input(shape=img_shape, name="noisy_image")
    t_input = layers.Input(shape=(), dtype=tf.int32, name="timestep")
    label_input = layers.Input(shape=(num_classes,), name="class_label")

    # Embed timestep t using sinusoidal embeddings
    def sinusoidal_embedding(timesteps, dim=128):
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = tf.cast(timesteps, tf.float32)[:, None] * tf.exp(-emb * tf.range(half_dim, dtype=tf.float32)[None, :])
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
        return emb

    t_emb = layers.Lambda(lambda t: sinusoidal_embedding(t, time_embedding_dim))(t_input)
    t_emb = layers.Dense(128, activation="relu")(t_emb)

    # Concatenate label with time embedding
    class_time_emb = layers.Concatenate()([label_input, t_emb])
    class_time_emb = layers.Dense(np.prod(img_shape), activation="relu")(class_time_emb)
    class_time_emb = layers.Reshape(img_shape)(class_time_emb)

    # Concatenate noisy image with conditional embedding
    x = layers.Concatenate()([img_input, class_time_emb])

    # Convolutional block
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(1, 3, padding="same")(x)  # Predict noise

    return models.Model(inputs=[img_input, t_input, label_input], outputs=x, name="DenoisingModel")


# -------------------------------------
# SECTION 6: Training Step and Loss Function
# -------------------------------------

# Mean Squared Error loss for noise prediction
mse_loss = tf.keras.losses.MeanSquaredError()

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

@tf.function
def train_step(model, x_0, y_class, timesteps, alpha_hat):
    """
    Perform one training step of the diffusion model.
    Args:
        model: Denoising network.
        x_0: Original images (batch).
        y_class: One-hot labels.
        timesteps: Random timesteps for each image.
        alpha_hat: Precomputed ᾱ_t for all timesteps.
    Returns:
        loss value
    """
    # Add noise to x_0 based on timestep t
    noise = tf.random.normal(shape=tf.shape(x_0))
    sqrt_alpha_hat = tf.sqrt(tf.gather(alpha_hat, timesteps))
    sqrt_one_minus_alpha_hat = tf.sqrt(1.0 - tf.gather(alpha_hat, timesteps))

    # Reshape for broadcasting
    sqrt_alpha_hat = tf.reshape(sqrt_alpha_hat, (-1, 1, 1, 1))
    sqrt_one_minus_alpha_hat = tf.reshape(sqrt_one_minus_alpha_hat, (-1, 1, 1, 1))

    x_t = sqrt_alpha_hat * x_0 + sqrt_one_minus_alpha_hat * noise

    with tf.GradientTape() as tape:
        noise_pred = model([x_t, timesteps, y_class], training=True)
        loss = mse_loss(noise, noise_pred)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# -------------------------------------
# SECTION 7: Full Training Loop (with Logging and Early Stopping)
# -------------------------------------
def train_diffusion_model(model, train_dataset, val_dataset, alpha_hat, epochs, summary_writer, patience=10):
    log_message("Starting diffusion model training with early stopping...")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in trange(epochs, desc="Epochs"):
        train_losses = []

        for x_batch, y_batch in train_dataset:
            batch_size = tf.shape(x_batch)[0]
            t = tf.random.uniform((batch_size,), minval=0, maxval=alpha_hat.shape[0], dtype=tf.int32)
            loss = train_step(model, x_batch, y_batch, t, alpha_hat)
            train_losses.append(loss.numpy())

        avg_train_loss = np.mean(train_losses)

        # Validation loss
        val_losses = []
        for val_x, val_y in val_dataset:
            batch_size = tf.shape(val_x)[0]
            t = tf.random.uniform((batch_size,), minval=0, maxval=alpha_hat.shape[0], dtype=tf.int32)
            val_loss = train_step(model, val_x, val_y, t, alpha_hat)  # We reuse train_step without backprop
            val_losses.append(val_loss.numpy())
        avg_val_loss = np.mean(val_losses)

        log_message(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        with summary_writer.as_default():
            tf.summary.scalar("Train Loss", avg_train_loss, step=epoch)
            tf.summary.scalar("Val Loss", avg_val_loss, step=epoch)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            model.save_weights(os.path.join(CHECKPOINT_DIR, "diffusion_best.h5"))
        else:
            patience_counter += 1
            log_message(f"Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                log_message("Early stopping triggered.")
                break

        # Optional periodic saving
        if (epoch + 1) % 50 == 0:
            model.save_weights(os.path.join(CHECKPOINT_DIR, f"diffusion_epoch_{epoch+1}.h5"))


# -------------------------------------
# SECTION 8: Sampling and Denoising (Reverse Diffusion Process)
# -------------------------------------
def sample_from_diffusion(model, alpha_hat, timesteps, num_classes, img_shape, n_samples=9):
    log_message("Sampling synthetic images using reverse diffusion...")

    # Start with pure Gaussian noise
    x = tf.random.normal((n_samples, *img_shape))

    # Create label embeddings for each class
    labels = np.tile(np.arange(num_classes), n_samples // num_classes + 1)[:n_samples]
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes)

    # Reverse diffusion
    for t in reversed(range(timesteps)):

        t_tensor = tf.convert_to_tensor(np.full(n_samples, t), dtype=tf.int32)
        z_t = model([x, t_tensor, one_hot_labels], training=False)

        alpha = tf.gather(alpha_hat, t)
        alpha = tf.reshape(alpha, (-1, 1, 1, 1))

        noise_term = tf.random.normal(tf.shape(x)) if t > 0 else 0.0
        beta_t = 1 - alpha
        x = (1 / tf.sqrt(alpha)) * (x - beta_t / tf.sqrt(1 - alpha) * z_t) + tf.sqrt(beta_t) * noise_term

    x = tf.clip_by_value(x, -1.0, 1.0)
    return x.numpy(), labels

# -------------------------------------
# SECTION 9: Generate and Visualize Conditional Samples
# -------------------------------------
def generate_and_plot_diffusion_samples(model, alpha_hat, timesteps, img_shape, num_classes, save_path=None):
    log_message("Generating conditional diffusion samples...")

    # Generate one sample per class
    n_samples = num_classes
    x = tf.random.normal((n_samples, *img_shape))

    labels = np.arange(num_classes)
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes)

    # Reverse diffusion
    for t in reversed(range(timesteps)):
        t_tensor = tf.convert_to_tensor([t] * n_samples)
        noise_pred = model([x, t_tensor, one_hot_labels], training=False)

        alpha_t = tf.gather(alpha_hat, t)
        alpha_t = tf.reshape(alpha_t, (-1, 1, 1, 1))

        beta_t = 1.0 - alpha_t
        if t > 0:
            noise = tf.random.normal(tf.shape(x))
        else:
            noise = 0.0

        x = (1 / tf.sqrt(alpha_t)) * (x - beta_t / tf.sqrt(1.0 - alpha_t) * noise_pred) + tf.sqrt(beta_t) * noise

    # Postprocess and plot
    x = tf.clip_by_value(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0  # scale to [0, 1]

    plt.figure(figsize=(12, 2))
    for i in range(num_classes):
        plt.subplot(1, num_classes, i + 1)
        plt.imshow(x[i].numpy().squeeze(), cmap="gray")
        plt.title(f"Class {i}")
        plt.axis("off")

    if save_path:
        plt.savefig(save_path)
    plt.close()


# -------------------------------------
# SECTION 10: Compute FID Score
# -------------------------------------
def calculate_fid(real_images, generated_images):
    """
    Computes FID score between real and generated grayscale images.
    Both inputs must be in range [0, 1] and shape [N, H, W, 1].
    """
    # Resize to 299x299 and convert grayscale to RGB
    real_resized = tf.image.resize(real_images, (299, 299))
    fake_resized = tf.image.resize(generated_images, (299, 299))

    real_rgb = tf.image.grayscale_to_rgb(real_resized)
    fake_rgb = tf.image.grayscale_to_rgb(fake_resized)

    # Preprocess for InceptionV3
    real_rgb = preprocess_input(real_rgb)
    fake_rgb = preprocess_input(fake_rgb)

    # Load InceptionV3
    inception = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    # Get activations
    act1 = inception.predict(real_rgb, verbose=0)
    act2 = inception.predict(fake_rgb, verbose=0)

    # Calculate statistics
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)

    # Compute Fréchet distance
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1 @ sigma2)

    # Numerical stability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    return fid


# -------------------------------------
# SECTION 11: CNN Classifier for Evaluation
# -------------------------------------
def build_classifier(input_shape=(40, 40, 1), num_classes=9):
    model = models.Sequential(name="Diffusion_Eval_Classifier")
    model.add(layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# -------------------------------------
# SECTION 12: Train & Evaluate Real-Only Baseline Classifier
# -------------------------------------
def train_real_classifier(train_data, val_data, input_shape, num_classes, checkpoint_path, epochs=30):
    log_message("Training real-only CNN classifier for evaluation...")

    classifier = build_classifier(input_shape, num_classes)

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    )

    history = classifier.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[checkpoint_cb, early_stop_cb],
        verbose=2
    )

    log_message("Classifier training complete.")
    return classifier


# -------------------------------------
# SECTION 13: Generate Synthetic Samples per Class
# -------------------------------------
def generate_synthetic_per_class(model, alpha_hat, timesteps, img_shape, num_classes, samples_per_class=100, save_dir=SYNTHETIC_DIR):
    log_message(f"Generating {samples_per_class} synthetic samples for each of {num_classes} classes...")

    os.makedirs(save_dir, exist_ok=True)

    for class_id in range(num_classes):
        class_dir = os.path.join(save_dir, f"class_{class_id}")
        os.makedirs(class_dir, exist_ok=True)

        for i in range(samples_per_class):
            print(f"[INFO] Generating sample {i + 1}/{samples_per_class} for class {class_id}")

            label = tf.keras.utils.to_categorical([class_id], num_classes=num_classes)
            x = tf.random.normal((1, *img_shape))

            # Reverse diffusion sampling
            for t in reversed(range(timesteps)):
                t_tensor = tf.convert_to_tensor([t])
                z_t = model([x, t_tensor, label], training=False)

                alpha = tf.gather(alpha_hat, t)
                alpha = tf.reshape(alpha, (1, 1, 1, 1))

                noise_term = tf.random.normal(tf.shape(x)) if t > 0 else 0.0
                beta_t = 1 - alpha
                x = (1 / tf.sqrt(alpha)) * (x - beta_t / tf.sqrt(1 - alpha) * z_t) + tf.sqrt(beta_t) * noise_term

            # Post-process and save
            sample = tf.clip_by_value(x, -1.0, 1.0)
            sample = (sample + 1.0) / 2.0  # Rescale to [0, 1]
            sample = tf.squeeze(sample).numpy()

            save_path = os.path.join(class_dir, f"sample_{i}.npy")
            np.save(save_path, sample)

    log_message("Completed generation of all synthetic class samples.")


# -------------------------------------
# SECTION 14: Load Synthetic Samples from Disk
# -------------------------------------
def load_synthetic_samples(root_dir, num_classes, samples_per_class=100, img_shape=(40, 40, 1)):
    log_message(f"Loading synthetic samples from '{root_dir}'...")

    x_synthetic = []
    y_synthetic = []

    for class_id in range(num_classes):
        class_dir = os.path.join(root_dir, f"class_{class_id}")
        if not os.path.exists(class_dir):
            log_message(f"Warning: Class folder '{class_dir}' not found. Skipping...", display=True)
            continue

        sample_files = sorted(os.listdir(class_dir))[:samples_per_class]
        for filename in sample_files:
            file_path = os.path.join(class_dir, filename)
            img = np.load(file_path).reshape(img_shape)
            x_synthetic.append(img)
            y_synthetic.append(class_id)

    x_synthetic = np.array(x_synthetic, dtype=np.float32)
    y_synthetic = np.array(y_synthetic)
    y_synthetic_onehot = tf.keras.utils.to_categorical(y_synthetic, num_classes)

    log_message(f"Loaded {len(x_synthetic)} synthetic samples.")

    return x_synthetic, y_synthetic_onehot


# -------------------------------------
# SECTION 16: Train Classifier on Real Data Only
# -------------------------------------
def train_real_only_classifier(train_data, val_data, input_shape, num_classes, epochs=20, log_path="classifier_real_only.h5"):
    log_message("Training classifier on real data only...")

    classifier = build_classifier(input_shape=input_shape, num_classes=num_classes)

    history = classifier.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        verbose=2
    )

    classifier.save(log_path)

    log_message(f"Saved real-only classifier to {log_path}")

    return classifier


# -------------------------------------
# SECTION 17: Evaluate Classifier on Real vs. Real+Synthetic
# -------------------------------------
def evaluate_classifier_on_real_vs_combined(classifier_real, classifier_combined, x_test, y_test):
    """
    Compare classifiers trained on real-only vs. real+synthetic data.
    """
    log_message("Evaluating classifier trained on real data only...")
    real_eval = classifier_real.evaluate(x_test, y_test, verbose=0)
    real_accuracy = real_eval[1]  # Assuming metrics=["accuracy"]

    log_message("Evaluating classifier trained on real + synthetic data...")
    combined_eval = classifier_combined.evaluate(x_test, y_test, verbose=0)
    combined_accuracy = combined_eval[1]

    log_message(f"Real-Only Accuracy: {real_accuracy:.4f}")
    log_message(f"Real+Synthetic Accuracy: {combined_accuracy:.4f}")

    return {
        "real_only": real_accuracy,
        "real_plus_synthetic": combined_accuracy
    }


# -------------------------------------
# SECTION 18: Log Comparison Results
# -------------------------------------
def log_comparison_results(results_dict, output_path="comparison_results.txt"):
    """
    Save performance comparison of real-only vs. real+synthetic classifiers.
    """
    log_message("Logging classifier comparison results...")

    with open(output_path, "w") as f:
        f.write("=== Classifier Evaluation Results ===\n")
        f.write(f"Real-Only Accuracy:        {results_dict['real_only']:.4f}\n")
        f.write(f"Real+Synthetic Accuracy:   {results_dict['real_plus_synthetic']:.4f}\n")

    log_message(f"Results written to {output_path}")


# -------------------------------------
# SECTION 19: Main Runner Script
# -------------------------------------
if __name__ == "__main__":
    log_message("Initializing Conditional Diffusion Model...")

    # Load or create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(BATCH_SIZE)

    # Build model
    diffusion_model = build_denoising_model(IMG_SHAPE, NUM_CLASSES)

    if RUN_MODE == "train":
        # Train diffusion model
        train_diffusion_model(
            model=diffusion_model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            alpha_hat=alphas_cumprod_tf,
            epochs=EPOCHS,
            summary_writer=summary_writer
        )

        # Save final model
        diffusion_model.save_weights(os.path.join(CHECKPOINT_DIR, "diffusion_final.h5"))

    elif RUN_MODE == "eval_only":
        log_message("Loading best saved model for evaluation...")
        diffusion_model.load_weights(os.path.join(CHECKPOINT_DIR, "diffusion_best.h5"))

    # Generate and plot sample grid
    generate_and_plot_diffusion_samples(
        model=diffusion_model,
        alpha_hat=alphas_cumprod_tf,
        timesteps=T,
        img_shape=IMG_SHAPE,
        num_classes=NUM_CLASSES
    )

    # Compute FID
    real_images, _ = next(iter(val_dataset))
    fake_images, _ = sample_from_diffusion(
        model=diffusion_model,
        alpha_hat=alphas_cumprod_tf,
        timesteps=T,
        num_classes=NUM_CLASSES,
        img_shape=IMG_SHAPE,
        n_samples=real_images.shape[0]
    )

    fid_score = calculate_fid(real_images, fake_images)

    log_message(f"FID Score on Validation Set: {fid_score:.4f}")
