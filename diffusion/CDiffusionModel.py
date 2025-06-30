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
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping


# ==============================
# Set Random Seeds
# ==============================
def set_random_seeds(seed=42):
    """Ensure reproducibility by fixing random seeds."""
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
# Load Hyperparameters from Config
# ==============================
IMG_SHAPE = tuple(config["IMG_SHAPE"])
NUM_CLASSES = config["NUM_CLASSES"]
BATCH_SIZE = config["BATCH_SIZE"]
EPOCHS = config["EPOCHS"]
LR = config["LR"]
BETA_1 = config["BETA_1"]
RUN_MODE = config.get("mode", "train")  # 'train' or 'eval_only'
VERBOSE = config.get("verbose", True)
T = config.get("T", 1000)  # Total diffusion timesteps
TIMESTEPS = config.get("TIMESTEPS", 1000)
LATENT_DIM = config.get("LATENT_DIM", 100)

# ==============================
# Load Patience Parameter
# ==============================
PATIENCE = config.get("early_stopping", {}).get("patience", config.get("training", {}).get("patience", 10))

# ==============================
# Define Paths for Output
# ==============================
EXPERIMENT_NAME = "conditional_diffusion"
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
    """Print and log messages with timestamps."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    if VERBOSE and display:
        print(full_msg)
    with open(LOG_FILE, "a") as f:
        f.write(full_msg + "\n")

# ==============================
# Create Output Directories
# ==============================
for dir_path in [CHECKPOINT_DIR, SYNTHETIC_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ==============================
# TensorBoard Logger
# ==============================
summary_writer = tf.summary.create_file_writer(TENSORBOARD_DIR)

# ==============================
# Fixed Labels for Visualization
# ==============================
FIXED_LABELS = np.arange(NUM_CLASSES).reshape(-1, 1)
FIXED_LABELS_ONEHOT = tf.keras.utils.to_categorical(FIXED_LABELS, NUM_CLASSES)


# =====================================
# SECTION 2A: Data Loading and Preprocessing
# =====================================

def load_malware_dataset(data_path, img_shape, num_classes):
    """
    Load grayscale malware traffic images and one-hot encoded labels.
    - Normalizes pixel values to [0, 1]
    - Reshapes to match CNN input shape (H, W, C)
    - Converts class labels to one-hot encoding
    """
    x_train = np.load(os.path.join(data_path, "train_data.npy"))
    y_train = np.load(os.path.join(data_path, "train_labels.npy"))
    x_test = np.load(os.path.join(data_path, "test_data.npy"))
    y_test = np.load(os.path.join(data_path, "test_labels.npy"))

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # Reshape to (batch_size, H, W, C)
    x_train = np.reshape(x_train, (-1, *img_shape))
    x_test = np.reshape(x_test, (-1, *img_shape))

    # Convert labels to one-hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)

def create_datasets(batch_size):
    """
    Convert malware dataset into batched TensorFlow datasets.
    Splits test set in half for validation and test.
    Applies batching and prefetching for GPU efficiency.
    """
    (x_train, y_train), (x_test, y_test) = load_malware_dataset(DATA_PATH, IMG_SHAPE, NUM_CLASSES)

    # Split test set into validation and test
    val_split = len(x_test) // 2
    val_data = (x_test[:val_split], y_test[:val_split])
    test_data = (x_test[val_split:], y_test[val_split:])

    # Prepare tf.data.Dataset pipelines
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=10240).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices(val_data)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices(test_data)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, test_dataset


# =====================================
# SECTION 2B: Sinusoidal Time Embedding and alpha schedule function
# =====================================

def sinusoidal_embedding(timesteps, embedding_dim):
    """
    Applies sinusoidal positional encoding to scalar timestep input.

    Args:
        timesteps: Tensor of shape [batch_size], scalar timestep per sample
        embedding_dim: Size of the sinusoidal embedding

    Returns:
        Tensor of shape [batch_size, embedding_dim]
    """
    position = tf.cast(timesteps[:, tf.newaxis], tf.float32)  # [batch_size, 1]
    div_term = tf.exp(tf.range(0, embedding_dim, 2, dtype=tf.float32) * (-np.log(10000.0) / embedding_dim))

    sinusoid = tf.concat(
        [tf.sin(position * div_term), tf.cos(position * div_term)],
        axis=-1
    )  # Shape: [batch_size, embedding_dim]

    return sinusoid



def create_alpha_schedule(timesteps):
    """
    Creates the cumulative product of noise schedule (alpha_hat).
    This is commonly used in DDPM-based diffusion models.
    """
    beta_start = 0.0001
    beta_end = 0.02
    betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)
    alphas = 1.0 - betas
    alpha_hat = np.cumprod(alphas, axis=0)
    return alpha_hat


# =====================================
# SECTION 3: Noise Scheduler and Beta Schedule
# =====================================

class DiffusionScheduler:
    """
    Scheduler to manage beta values and compute q(x_t | x_0).
    This class precomputes diffusion constants and enables noise addition at any timestep t.
    """
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps

        # Linear beta schedule: beta_t increases from beta_start to beta_end
        self.betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)

        # Compute alpha_t and its cumulative product
        self.alphas = 1.0 - self.betas
        self.alpha_hat = np.cumprod(self.alphas, axis=0)

        # Precompute terms used during noise sampling
        self.sqrt_alpha_hat = np.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = np.sqrt(1.0 - self.alpha_hat)

    def q_sample(self, x_start, t, noise=None):
        """
        Diffusion forward process: q(x_t | x_0)
        Adds noise to clean image x_start at timestep t.

        Args:
            x_start: Clean input image tensor
            t: Timestep tensor (int32)
            noise: Optional external noise (defaults to random normal)
        Returns:
            Noisy image x_t at timestep t
        """
        if noise is None:
            noise = tf.random.normal(shape=tf.shape(x_start))

        # Gather precomputed constants at timestep t
        sqrt_alpha_hat_t = tf.gather(self.sqrt_alpha_hat, t)
        sqrt_one_minus_alpha_hat_t = tf.gather(self.sqrt_one_minus_alpha_hat, t)

        # Expand dimensions for broadcasting over image shape
        sqrt_alpha_hat_t = tf.reshape(sqrt_alpha_hat_t, (-1, 1, 1, 1))
        sqrt_one_minus_alpha_hat_t = tf.reshape(sqrt_one_minus_alpha_hat_t, (-1, 1, 1, 1))

        # Compute x_t using q(x_t | x_0)
        return sqrt_alpha_hat_t * x_start + sqrt_one_minus_alpha_hat_t * noise


# =====================================
# SECTION 4: Define the Conditional UNet Model (Noise Predictor)
# =====================================

def get_timestep_embedding(timesteps, embedding_dim=128):
    """
    Compute sinusoidal timestep embeddings.
    These help the model understand the current diffusion timestep.

    Args:
        timesteps: Tensor of shape (batch_size,) with timestep integers
        embedding_dim: Dimensionality of the sinusoidal embedding
    Returns:
        Tensor of shape (batch_size, embedding_dim)
    """
    half_dim = embedding_dim // 2
    exponent = -np.log(10000) / (half_dim - 1)
    emb = tf.cast(timesteps, tf.float32)[:, None] * tf.exp(tf.range(half_dim, dtype=tf.float32) * exponent)
    emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
    return emb

def build_conditional_unet(img_shape, num_classes, embedding_dim=128):
    """
    Build a small UNet-like CNN that conditions on both timestep and class label.

    Args:
        img_shape: Shape of input image (H, W, C)
        num_classes: Number of class labels for conditional generation
        embedding_dim: Size of timestep embedding vector
    Returns:
        Keras Model that predicts noise given noisy image, timestep, and class label
    """
    # Inputs: noisy image, timestep, and one-hot label
    image_input = layers.Input(shape=img_shape, name="noisy_input")
    timestep_input = layers.Input(shape=(), dtype=tf.int32, name="timestep_input")
    label_input = layers.Input(shape=(num_classes,), name="label_input")

    # --- Timestep Embedding ---
    t_embedding = get_timestep_embedding(timestep_input, embedding_dim)
    t_dense = layers.Dense(np.prod(img_shape), activation="relu")(t_embedding)
    t_reshaped = layers.Reshape(img_shape)(t_dense)

    # --- Label Embedding ---
    y_dense = layers.Dense(np.prod(img_shape), activation="relu")(label_input)
    y_reshaped = layers.Reshape(img_shape)(y_dense)

    # --- Combine Inputs ---
    x = layers.Concatenate()([image_input, t_reshaped, y_reshaped])

    # --- Downsampling ---
    x = layers.Conv2D(64, 3, activation='relu', padding="same")(x)
    x = layers.MaxPooling2D()(x)

    # --- Bottleneck ---
    x = layers.Conv2D(128, 3, activation='relu', padding="same")(x)

    # --- Upsampling ---
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding="same")(x)

    # --- Output Layer ---
    x = layers.Conv2D(img_shape[-1], 1, activation=None)(x)  # Predict noise

    return models.Model(inputs=[image_input, timestep_input, label_input], outputs=x, name="Conditional_UNet")


# =====================================
# SECTION 5: Loss Function and Optimizer
# =====================================

# ------------------------------------------------------
# Define Mean Squared Error (MSE) loss for noise prediction
# This measures how close the model's predicted noise is to the actual noise
# ------------------------------------------------------
mse_loss = tf.keras.losses.MeanSquaredError()

# ------------------------------------------------------
# Optimizer: Adam (can be swapped with legacy Adam if needed for Apple Silicon)
# Learning rate and beta1 are loaded from config.yaml
# ------------------------------------------------------
optimizer = tf.keras.optimizers.Adam(learning_rate=LR, beta_1=BETA_1)


# =====================================
# SECTION 6A: Noise Schedule and Diffusion Utilities
# =====================================

def get_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    """
    Create a linear beta schedule for the diffusion process.

    Args:
        T: Total number of diffusion steps
        beta_start: Starting beta value (small noise)
        beta_end: Ending beta value (larger noise)
    Returns:
        Numpy array of shape (T,) containing beta values
    """
    return np.linspace(beta_start, beta_end, T, dtype=np.float32)

# -------------------------------------------------------
# Initialize beta schedule and derive alpha-related constants
# These are used for both forward (q(x_t | x_0)) and reverse process
# -------------------------------------------------------
betas = get_beta_schedule(T)
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas)

# Convert constants to TensorFlow tensors
betas_tf = tf.constant(betas, dtype=tf.float32)
alphas_cumprod_tf = tf.constant(alphas_cumprod, dtype=tf.float32)
sqrt_alphas_cumprod = tf.sqrt(alphas_cumprod_tf)
sqrt_one_minus_alphas_cumprod = tf.sqrt(1.0 - alphas_cumprod_tf)

def add_noise(x_start, noise, t):
    """
    Apply noise to clean image x_start at timestep t: q(x_t | x_0)

    Args:
        x_start: Tensor of clean images
        noise: Random noise tensor
        t: Timestep tensor (int32) of shape (batch_size,)
    Returns:
        Noisy image x_t
    """
    batch_size = tf.shape(x_start)[0]

    # Gather time-dependent coefficients
    sqrt_alpha_cumprod_t = tf.gather(sqrt_alphas_cumprod, t)
    sqrt_one_minus_alpha_cumprod_t = tf.gather(sqrt_one_minus_alphas_cumprod, t)

    # Reshape for broadcasting
    sqrt_alpha_cumprod_t = tf.reshape(sqrt_alpha_cumprod_t, [batch_size, 1, 1, 1])
    sqrt_one_minus_alpha_cumprod_t = tf.reshape(sqrt_one_minus_alpha_cumprod_t, [batch_size, 1, 1, 1])

    # Compute x_t
    return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise


# =====================================
# SECTION 6B: Build Conditional Diffusion Model (UNet)
# =====================================

def build_conditional_diffusion_model(img_shape, num_classes):
    """
    Builds the UNet-based conditional diffusion model.

    Args:
        img_shape: Shape of input image (e.g., (40, 40, 1))
        num_classes: Number of class labels (for conditional generation)

    Returns:
        A Keras Model ready for training.
    """

    # Inputs
    x_input = tf.keras.Input(shape=img_shape, name="x_input")              # Input image
    t_input = tf.keras.Input(shape=(), name="timesteps", dtype=tf.int32)  # ✅ Scalar input
    y_input = tf.keras.Input(shape=(num_classes,), name="y_input")         # One-hot label

    # -----------------------------------------------------
    # Time Embedding: sinusoidal function of timestep t
    # -----------------------------------------------------
    time_embedding_dim = 64
    t_emb = tf.keras.layers.Lambda(lambda t: sinusoidal_embedding(t, time_embedding_dim))(t_input)

    t_emb = tf.keras.layers.Dense(128, activation="swish")(t_emb)
    t_emb = tf.keras.layers.Dense(128, activation="swish")(t_emb)

    # -----------------------------------------------------
    # Label Embedding (one-hot input) + Dense layers
    # -----------------------------------------------------
    y_emb = tf.keras.layers.Dense(128, activation="swish")(y_input)
    y_emb = tf.keras.layers.Dense(128, activation="swish")(y_emb)

    # -----------------------------------------------------
    # Prepare Broadcasted Embeddings
    # -----------------------------------------------------
    t_broadcast = tf.keras.layers.Lambda(lambda t: tf.expand_dims(tf.expand_dims(t, 1), 1))(t_emb)
    t_broadcast = tf.keras.layers.Lambda(lambda t: tf.tile(t, [1, img_shape[0], img_shape[1], 1]))(t_broadcast)

    y_broadcast = tf.keras.layers.Lambda(lambda y: tf.expand_dims(tf.expand_dims(y, 1), 1))(y_emb)
    y_broadcast = tf.keras.layers.Lambda(lambda y: tf.tile(y, [1, img_shape[0], img_shape[1], 1]))(y_broadcast)

    # -----------------------------------------------------
    # Input Processing + Concatenate Embeddings
    # -----------------------------------------------------
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="swish")(x_input)
    x = tf.keras.layers.Concatenate()([x, t_broadcast, y_broadcast])

    # -----------------------------------------------------
    # Mini UNet-like Block (simplified)
    # -----------------------------------------------------
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="swish")(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="swish")(x)
    output = tf.keras.layers.Conv2D(img_shape[-1], 1, padding="same")(x)  # Output: predicted noise

    return tf.keras.Model(inputs=[x_input, y_input, t_input], outputs=output)


# =====================================
# SECTION 7: Define the Denoising Network (U-Net Inspired)
# =====================================

def build_denoising_model(img_shape, num_classes, time_embedding_dim=128):
    """
    Build a CNN that predicts noise from a noisy image,
    while conditioning on timestep and class label.

    Args:
        img_shape: Input image shape (H, W, C)
        num_classes: Number of class labels (for one-hot conditioning)
        time_embedding_dim: Size of the timestep embedding vector

    Returns:
        Keras model that maps (noisy_image, timestep, class_label) → predicted noise
    """
    # Inputs
    img_input = layers.Input(shape=img_shape, name="noisy_image")
    t_input = layers.Input(shape=(), dtype=tf.int32, name="timestep")
    label_input = layers.Input(shape=(num_classes,), name="class_label")

    # Apply sinusoidal embedding to t_input
    # t_emb = layers.Lambda(lambda t: sinusoidal_embedding(t, time_embedding_dim))(t_input) # no squeeze, no expand_dims
    t_emb = layers.Lambda(lambda t: sinusoidal_embedding(tf.reshape(t, [-1]), time_embedding_dim))(t_input)
    t_emb = layers.Dense(128, activation="relu")(t_emb)  # Project embedding

    # --------------------------------------
    # Combine label and time embeddings
    # --------------------------------------
    class_time_emb = layers.Concatenate()([label_input, t_emb])
    class_time_emb = layers.Dense(np.prod(img_shape), activation="relu")(class_time_emb)
    class_time_emb = layers.Reshape(img_shape)(class_time_emb)

    # --------------------------------------
    # Concatenate noisy image with embedding
    # --------------------------------------
    x = layers.Concatenate()([img_input, class_time_emb])

    # --------------------------------------
    # Convolutional Backbone (simple U-Net like)
    # --------------------------------------
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(1, 3, padding="same")(x)  # Output: predicted noise (same shape as input)

    return models.Model(
        inputs=[img_input, label_input, t_input],
        outputs=x,
        name="DenoisingModel"
    )


# --------------------------------------
# Sinusoidal Embedding for timestep t
# --------------------------------------

def sinusoidal_embedding(t, dim):
    """
        Safer sinusoidal embedding for arbitrary batch sizes and scalar timestep input.
        Args:
            t: Tensor of shape (batch_size,) or (batch_size, 1)
            dim: Output embedding size
        Returns:
            Tensor of shape (batch_size, dim)
    """
    t = tf.reshape(t, [-1])  # (batch_size,)
    t = tf.cast(t, tf.float32)
    half_dim = dim // 2
    emb = tf.math.log(10000.0) / (tf.cast(half_dim, tf.float32) - 1)
    emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
    emb = tf.expand_dims(t, -1) * tf.expand_dims(emb, 0)
    emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)

    return emb  # (batch_size, dim)

# =====================================
# SECTION 8: Training Step and Loss Function
# =====================================

# -----------------------------------------------
# Define the Mean Squared Error loss for training
# Repeated here for compatibility in other modules
# -----------------------------------------------
mse_loss = tf.keras.losses.MeanSquaredError()

# -----------------------------------------------
# Define the Adam optimizer (configurable via config.yaml)
# -----------------------------------------------
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

@tf.function
def train_step(model, x_0, y_class, t, alpha_hat):
    """
    Performs one training step for the diffusion model.
    This trains the model to predict the noise added to a clean input x_0.

    Args:
        model: The denoising model (predicts noise from x_t)
        x_0: Original clean images (batch)
        y_class: One-hot encoded class labels
        timesteps: Randomly sampled diffusion timesteps (shape: [batch_size])
        alpha_hat: Tensor of shape (T,) with cumulative alpha values

    Returns:
        Scalar loss value
    """
    # Sample random noise to add to clean images
    noise = tf.random.normal(shape=tf.shape(x_0))

    # Gather sqrt(alphâ_t) and sqrt(1 - alphâ_t) for each timestep
    sqrt_alpha_hat = tf.sqrt(tf.gather(alpha_hat, t))
    sqrt_one_minus_alpha_hat = tf.sqrt(1.0 - tf.gather(alpha_hat, t))

    # Reshape for broadcasting over image shape
    sqrt_alpha_hat = tf.reshape(sqrt_alpha_hat, (-1, 1, 1, 1))
    sqrt_one_minus_alpha_hat = tf.reshape(sqrt_one_minus_alpha_hat, (-1, 1, 1, 1))

    # Generate noisy input x_t from x_0 using q(x_t | x_0)
    x_t = sqrt_alpha_hat * x_0 + sqrt_one_minus_alpha_hat * noise

    # Forward pass through denoising model
    with tf.GradientTape() as tape:
        noise_pred = model([x_t, y_class, t], training=True)

        loss = mse_loss(noise, noise_pred)

    # Backpropagation and weight update
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


@tf.function
def eval_step(model, x_0, y_class, timesteps, alpha_hat):
    """Performs one forward pass without gradient update (for validation)."""
    noise = tf.random.normal(shape=tf.shape(x_0))

    batch_size = tf.shape(x_0)[0]
    timesteps = tf.reshape(timesteps, [-1])  # Ensure shape = [batch_size]

    # Apply forward process: q(x_t | x_0, t)
    sqrt_alpha_hat = tf.sqrt(tf.gather(alpha_hat, timesteps))
    sqrt_one_minus_alpha_hat = tf.sqrt(1.0 - tf.gather(alpha_hat, timesteps))
    sqrt_alpha_hat = tf.reshape(sqrt_alpha_hat, (-1, 1, 1, 1))
    sqrt_one_minus_alpha_hat = tf.reshape(sqrt_one_minus_alpha_hat, (-1, 1, 1, 1))
    x_t = sqrt_alpha_hat * x_0 + sqrt_one_minus_alpha_hat * noise

    # Predict noise
    noise_pred = model([x_t, y_class, timesteps], training=False)

    return mse_loss(noise, noise_pred)


# =====================================
# SECTION 9: Full Training Loop (with Logging and Early Stopping)
# =====================================

def train_diffusion_model(model, train_dataset, val_dataset, alpha_hat, epochs, summary_writer, checkpoint_dir, patience
):
    """
    Trains the conditional diffusion model using early stopping and logs metrics.

    Args:
        model: The denoising model to train
        train_dataset: tf.data.Dataset for training
        val_dataset: tf.data.Dataset for validation
        alpha_hat: Precomputed alphā_t values for all timesteps
        epochs: Maximum number of training epochs
        summary_writer: TensorBoard summary writer
        patience: Number of epochs to wait for improvement before stopping

    Returns:
        None (saves best model weights and logs losses)
    """
    log_message("Starting diffusion model training with early stopping...")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in trange(epochs, desc="Epochs"):
        train_losses = []

        # --- Training loop ---
        for x_batch, y_batch in train_dataset:
            batch_size = tf.shape(x_batch)[0]

            t = tf.random.uniform(shape=(batch_size,), minval=0, maxval=alpha_hat.shape[0], dtype=tf.int32)
            t = tf.reshape(t, [-1])  # Ensure shape = [batch_size]

            loss = train_step(model, x_batch, y_batch, t, alpha_hat)
            train_losses.append(loss.numpy())

        avg_train_loss = np.mean(train_losses)

        # --- Validation loop ---
        val_losses = []
        for val_x, val_y in val_dataset:
            batch_size = tf.shape(val_x)[0]
            t = tf.random.uniform((batch_size,), minval=0, maxval=alpha_hat.shape[0], dtype=tf.int32)
            val_loss = eval_step(model, val_x, val_y, t, alpha_hat)  # ✅ No gradients applied
            val_losses.append(val_loss.numpy())
        avg_val_loss = np.mean(val_losses)

        # --- Logging ---
        log_message(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        with summary_writer.as_default():
            tf.summary.scalar("Train Loss", avg_train_loss, step=epoch)
            tf.summary.scalar("Val Loss", avg_val_loss, step=epoch)

        # --- Early Stopping Check ---
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

        # --- Optional Periodic Saving ---
        if (epoch + 1) % 50 == 0:
            model.save_weights(os.path.join(CHECKPOINT_DIR, f"diffusion_epoch_{epoch+1}.h5"))


# =====================================
# SECTION 10: Sampling and Denoising (Reverse Diffusion Process)
# =====================================

def sample_from_diffusion(model, alpha_hat, timesteps, num_classes, img_shape, n_samples=9):
    """
    Generate synthetic samples by reverse diffusion using the trained model.

    Args:
        model: Trained denoising model
        alpha_hat: Precomputed ᾱ_t values for all timesteps
        timesteps: Total number of diffusion steps (T)
        num_classes: Number of conditional classes
        img_shape: Shape of generated image (H, W, C)
        n_samples: Total number of samples to generate

    Returns:
        x (numpy array): Generated synthetic images
        labels (numpy array): Corresponding class labels
    """
    log_message("Sampling synthetic images using reverse diffusion...")

    # Start from pure Gaussian noise
    x = tf.random.normal((n_samples, *img_shape))

    # Create balanced one-hot encoded labels for each class
    labels = np.tile(np.arange(num_classes), n_samples // num_classes + 1)[:n_samples]
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes)

    # Reverse diffusion process: x_T → x_0
    for t in reversed(range(timesteps)):
        # Use the same timestep t for the whole batch
        t_tensor = tf.convert_to_tensor(np.full(n_samples, t), dtype=tf.int32)

        # Predict noise at timestep t using the model
        z_t = model([x, t_tensor, one_hot_labels], training=False)

        # Gather ᾱ_t and compute update
        alpha = tf.gather(alpha_hat, t)
        alpha = tf.reshape(alpha, (-1, 1, 1, 1))

        # Add Gaussian noise during reverse steps (except at t = 0)
        noise_term = tf.random.normal(tf.shape(x)) if t > 0 else 0.0
        beta_t = 1 - alpha

        # DDPM sampling formula: reverse step
        x = (1 / tf.sqrt(alpha)) * (x - beta_t / tf.sqrt(1 - alpha) * z_t) + tf.sqrt(beta_t) * noise_term

    # Clamp to [-1, 1] to match original data range
    x = tf.clip_by_value(x, -1.0, 1.0)

    return x.numpy(), labels


# =====================================
# SECTION 11: Generate and Visualize Conditional Samples
# =====================================

def generate_and_plot_diffusion_samples(model, alpha_hat, timesteps, img_shape, num_classes, save_path=None):
    """
    Generate one sample per class and visualize them using reverse diffusion.

    Args:
        model: Trained denoising model
        alpha_hat: Precomputed ᾱ_t values for all timesteps
        timesteps: Total number of diffusion steps (T)
        img_shape: Shape of the image (H, W, C)
        num_classes: Total number of classes
        save_path: Optional file path to save the figure

    Returns:
        None (displays and optionally saves the figure)
    """
    log_message("Generating conditional diffusion samples...")

    # One sample for each class
    n_samples = num_classes
    x = tf.random.normal((n_samples, *img_shape))

    # Create one-hot encoded labels for each class
    labels = np.arange(num_classes)
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes)

    # Reverse diffusion process (sampling)
    for t in reversed(range(timesteps)):
        t_tensor = tf.convert_to_tensor(np.full(n_samples, t), dtype=tf.int32)

        # Predict noise and apply reverse step
        noise_pred = model([x, t_tensor, one_hot_labels], training=False)
        alpha_t = tf.gather(alpha_hat, t)
        alpha_t = tf.reshape(alpha_t, (-1, 1, 1, 1))
        beta_t = 1.0 - alpha_t

        # Add Gaussian noise except at step 0
        if t > 0:
            noise = tf.random.normal(tf.shape(x))
        else:
            noise = 0.0

        x = (1 / tf.sqrt(alpha_t)) * (x - beta_t / tf.sqrt(1.0 - alpha_t) * noise_pred) + tf.sqrt(beta_t) * noise

    # Rescale from [-1, 1] to [0, 1]
    x = tf.clip_by_value(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0

    # Plot each class sample
    plt.figure(figsize=(12, 2))
    for i in range(num_classes):
        plt.subplot(1, num_classes, i + 1)
        plt.imshow(x[i].numpy().squeeze(), cmap="gray")
        plt.title(f"Class {i}")
        plt.axis("off")

    if save_path:
        plt.savefig(save_path)
    plt.close()


# =====================================
# SECTION 12: Compute FID Score
# =====================================

def calculate_fid(real_images, generated_images):
    """
    Compute the Fréchet Inception Distance (FID) between real and generated images.

    Args:
        real_images: Real image tensor, shape (N, H, W, 1), values in [0, 1]
        generated_images: Generated image tensor, shape (N, H, W, 1), values in [0, 1]

    Returns:
        fid (float): FID score — lower is better
    """
    # Step 1: Resize images to 299x299 (required by InceptionV3)
    real_resized = tf.image.resize(real_images, (299, 299))
    fake_resized = tf.image.resize(generated_images, (299, 299))

    # Step 2: Convert grayscale to RGB
    real_rgb = tf.image.grayscale_to_rgb(real_resized)
    fake_rgb = tf.image.grayscale_to_rgb(fake_resized)

    # Step 3: Preprocess for InceptionV3 (same as ImageNet)
    real_rgb = preprocess_input(real_rgb)
    fake_rgb = preprocess_input(fake_rgb)

    # Step 4: Load InceptionV3 model for feature extraction (no top layer, avg pooling)
    inception = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    # Step 5: Compute activations
    act1 = inception.predict(real_rgb, verbose=0)
    act2 = inception.predict(fake_rgb, verbose=0)

    # Step 6: Calculate mean and covariance for both sets
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)

    # Step 7: Compute Fréchet distance
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1 @ sigma2)

    # Step 8: Remove imaginary component if present (numerical safety)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Final FID formula
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    return fid


# =====================================
# SECTION 13: CNN Classifier for Evaluation
# =====================================

def build_classifier(input_shape=(40, 40, 1), num_classes=9):
    """
    Build a simple CNN classifier to evaluate real and synthetic data.

    Args:
        input_shape: Shape of input images (H, W, C)
        num_classes: Number of class labels

    Returns:
        A compiled Keras model ready for training and evaluation
    """
    model = models.Sequential(name="Diffusion_Eval_Classifier")

    # Convolutional layers
    model.add(layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D())

    model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling2D())

    # Fully connected layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# =====================================
# SECTION 14: Train & Evaluate Real-Only Baseline Classifier
# =====================================

def train_real_classifier(train_data, val_data, input_shape, num_classes, checkpoint_path, epochs=30):
    """
    Train a CNN classifier using only real training data.

    Args:
        train_data: tf.data.Dataset for training
        val_data: tf.data.Dataset for validation
        input_shape: Shape of input images (H, W, C)
        num_classes: Number of output classes
        checkpoint_path: File path to save best model weights
        epochs: Number of training epochs

    Returns:
        Trained Keras classifier model
    """
    log_message("Training real-only CNN classifier for evaluation...")

    # Build model
    classifier = build_classifier(input_shape=input_shape, num_classes=num_classes)

    # Callbacks: checkpoint + early stopping
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

    # Train classifier
    history = classifier.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[checkpoint_cb, early_stop_cb],
        verbose=2
    )

    log_message("Classifier training complete.")

    return classifier


# =====================================
# SECTION 15: Generate Synthetic Samples per Class
# =====================================

def generate_synthetic_per_class(model, alpha_hat, timesteps, img_shape, num_classes, samples_per_class=100, save_dir=SYNTHETIC_DIR):
    """
    Generate and save synthetic samples for each class using reverse diffusion.

    Args:
        model: Trained diffusion model
        alpha_hat: Precomputed ᾱ_t values
        timesteps: Total diffusion steps (T)
        img_shape: Shape of the generated images (H, W, C)
        num_classes: Number of conditional classes
        samples_per_class: Number of samples to generate per class
        save_dir: Directory to store generated .npy samples

    Returns:
        None (saves files to disk)
    """
    log_message(f"Generating {samples_per_class} synthetic samples for each of {num_classes} classes...")
    os.makedirs(save_dir, exist_ok=True)

    for class_id in range(num_classes):
        class_dir = os.path.join(save_dir, f"class_{class_id}")
        os.makedirs(class_dir, exist_ok=True)

        for i in range(samples_per_class):
            print(f"[INFO] Generating sample {i + 1}/{samples_per_class} for class {class_id}")

            # label = tf.convert_to_tensor([class_id], dtype=tf.int32)  # shape (1,)
            label = tf.keras.utils.to_categorical([class_id], num_classes=num_classes).astype(np.float32)

            # Create random noise sample
            x = tf.random.normal((1, *img_shape))

            # Reverse diffusion process
            for t in reversed(range(timesteps)):
                # t_tensor = tf.convert_to_tensor([t], dtype=tf.int32)  # shape (1,)
                t_tensor = tf.convert_to_tensor([[t]], dtype=tf.int32)  # shape (1, 1)

                # z_t = model([x, t_tensor, label], training=False)
                z_t = model([x, label, t_tensor], training=False)

                alpha = tf.gather(alpha_hat, t)
                alpha = tf.reshape(alpha, (1, 1, 1, 1))
                beta_t = 1 - alpha
                noise_term = tf.random.normal(tf.shape(x)) if t > 0 else 0.0

                x = (1 / tf.sqrt(alpha)) * (x - beta_t / tf.sqrt(1 - alpha) * z_t) + tf.sqrt(beta_t) * noise_term

            # Rescale to [0, 1] and save
            sample = tf.clip_by_value(x, -1.0, 1.0)
            sample = (sample + 1.0) / 2.0
            sample = tf.squeeze(sample).numpy()

            save_path = os.path.join(class_dir, f"sample_{i}.npy")
            np.save(save_path, sample)

    log_message("Completed generation of all synthetic class samples.")


# =====================================
# SECTION 16: Load Synthetic Samples from Disk
# =====================================

def load_synthetic_samples(root_dir, num_classes, samples_per_class=100, img_shape=(40, 40, 1)):
    """
    Load synthetic samples for all classes from disk.

    Args:
        root_dir: Root directory containing per-class sample folders
        num_classes: Number of conditional classes
        samples_per_class: Number of .npy files to load per class
        img_shape: Shape of each image (H, W, C)

    Returns:
        x_synthetic: Array of synthetic images, shape (N, H, W, C)
        y_synthetic_onehot: One-hot encoded labels, shape (N, num_classes)
    """
    log_message(f"Loading synthetic samples from '{root_dir}'...")

    x_synthetic = []
    y_synthetic = []

    for class_id in range(num_classes):
        class_dir = os.path.join(root_dir, f"class_{class_id}")
        if not os.path.exists(class_dir):
            log_message(f"Warning: Class folder '{class_dir}' not found. Skipping...", display=True)
            continue

        # Load up to `samples_per_class` from this folder
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


# =====================================
# SECTION 17A: Train Classifier on Real Data Only
# =====================================

def train_real_only_classifier(train_data, val_data, input_shape, num_classes, epochs=20, log_path="classifier_real_only.h5"):
    """
    Train a CNN classifier using only real data and save the model.

    Args:
        train_data: tf.data.Dataset for training
        val_data: tf.data.Dataset for validation
        input_shape: Shape of the input image
        num_classes: Number of output classes
        epochs: Number of training epochs
        log_path: Path to save trained model weights

    Returns:
        classifier: Trained Keras model
    """
    log_message("Training classifier on real data only...")

    # Build classifier
    classifier = build_classifier(input_shape=input_shape, num_classes=num_classes)

    # Train model
    history = classifier.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        verbose=2
    )

    # Save model weights
    classifier.save(log_path)
    log_message(f"Saved real-only classifier to {log_path}")

    return classifier, history


# =====================================
# SECTION 17B: Train Classifier on Real plus Synthetic
# =====================================

def train_classifier_on_real_plus_synthetic(x_real, y_real, x_synth, y_synth, img_shape, num_classes, batch_size=64, epochs=20, log_dir="logs/classifier_combined", save_path="classifier_combined_diffusion.h5"):
    """
    Train a classifier on both real and synthetic samples.

    Args:
        x_real (np.ndarray): Real images [N, H, W, C]
        y_real (np.ndarray): One-hot real labels [N, num_classes]
        x_synth (np.ndarray): Synthetic images [M, H, W, C]
        y_synth (np.ndarray): One-hot synthetic labels [M, num_classes]
        img_shape (tuple): Input image shape
        num_classes (int): Number of classes
        batch_size (int): Batch size for training
        epochs (int): Number of epochs
        log_dir (str): Directory to store TensorBoard logs

    Returns:
        model: Trained classifier model
        history: Training history for plotting
    """

    # Combine and shuffle
    x_all = np.concatenate([x_real, x_synth], axis=0)
    y_all = np.concatenate([y_real, y_synth], axis=0)
    indices = np.random.permutation(len(x_all))
    x_all, y_all = x_all[indices], y_all[indices]

    # Train-validation split (80-20)
    split = int(0.8 * len(x_all))
    x_train, y_train = x_all[:split], y_all[:split]
    x_val, y_val = x_all[split:], y_all[split:]

    # Build classifier model
    model = build_classifier(input_shape=img_shape, num_classes=num_classes)

    # Set up callbacks
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_cb = TensorBoard(log_dir=os.path.join(log_dir, timestamp))
    earlystop_cb = EarlyStopping(patience=5, restore_best_weights=True)

    # Train model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[tensorboard_cb, earlystop_cb],
        verbose=2
    )

    model.save(save_path)  # Save final model

    return model, history


# =====================================
# SECTION 18: Evaluate Classifier on Real vs. Real+Synthetic
# =====================================

def evaluate_classifier_on_real_vs_combined(classifier_real, classifier_combined, x_test, y_test):
    """
    Evaluate and compare classifiers trained on:
    - Real-only data
    - Real + synthetic data

    Args:
        classifier_real: Model trained only on real data
        classifier_combined: Model trained on real + synthetic data
        x_test: Test set inputs (real)
        y_test: One-hot encoded test labels

    Returns:
        Dictionary with real-only and real+synthetic accuracies
    """
    log_message("Evaluating classifier trained on real data only...")
    real_eval = classifier_real.evaluate(x_test, y_test, verbose=0)
    real_accuracy = real_eval[1]  # accuracy is at index 1

    log_message("Evaluating classifier trained on real + synthetic data...")
    combined_eval = classifier_combined.evaluate(x_test, y_test, verbose=0)
    combined_accuracy = combined_eval[1]

    # Log and return results
    log_message(f"Real-Only Accuracy: {real_accuracy:.4f}")
    log_message(f"Real+Synthetic Accuracy: {combined_accuracy:.4f}")

    return {
        "real_only": real_accuracy,
        "real_plus_synthetic": combined_accuracy
    }


# =====================================
# SECTION 19: Log Comparison Results
# =====================================

def log_comparison_results(results_dict, output_path="comparison_results.txt"):
    """
    Save the accuracy comparison between real-only and real+synthetic classifiers to disk.

    Args:
        results_dict: Dictionary containing evaluation metrics
            e.g., {"real_only": 0.89, "real_plus_synthetic": 0.92}
        output_path: File path to save the comparison results

    Returns:
        None (writes results to .txt file)
    """
    log_message("Logging classifier comparison results...")

    with open(output_path, "w") as f:
        f.write("=== Classifier Evaluation Results ===\n")
        f.write(f"Real-Only Accuracy:        {results_dict['real_only']:.4f}\n")
        f.write(f"Real+Synthetic Accuracy:   {results_dict['real_plus_synthetic']:.4f}\n")

    log_message(f"Results written to {output_path}")


# =====================================
# SECTION 20: Save Training & Generation Plots
# =====================================

def save_metric_plot(history, metrics, title, ylabel, filename):
    """
    Save a plot for specified training/validation metrics from Keras History.

    Args:
        history: Keras History object (returned from model.fit()).
        metrics: List of metric names to plot (e.g., ["loss", "val_loss"]).
        title: Plot title (e.g., "Classifier Accuracy").
        ylabel: Label for the y-axis (e.g., "Accuracy", "Loss").
        filename: Output file name (e.g., "accuracy_plot.png").
    """
    plt.figure(figsize=(8, 5))
    for metric in metrics:
        if metric in history.history:
            plt.plot(history.history[metric], label=metric.replace("_", " ").title())

    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_model_comparison(metrics_dict, ylabel, title, filename):
    """
    Plot and save a bar chart comparing metrics (e.g., accuracy or FID) across multiple models.

    Args:
        metrics_dict: Dictionary with model names as keys and float metric values (e.g., {"Diffusion": 0.91, "Autoregressive": 0.87})
        ylabel: Label for the y-axis (e.g., "Accuracy", "FID Score")
        title: Plot title (e.g., "Model Accuracy Comparison")
        filename: File name to save the plot (e.g., "model_accuracy_comparison.png")
    """
    models = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    plt.figure(figsize=(10, 5))
    bars = plt.bar(models, values)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{height:.4f}",
                 ha='center', va='bottom', fontsize=10)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# =====================================
# SECTION 21: Main Runner Script – Conditional Diffusion Model
# =====================================

if __name__ == "__main__":
    log_message("Initializing Conditional Diffusion Model...")

    # -----------------------------------------------------
    # Load training, validation, and test datasets
    # -----------------------------------------------------
    train_dataset, val_dataset, test_dataset = create_datasets(BATCH_SIZE)

    # -----------------------------------------------------
    # Build the Conditional Diffusion Model
    # -----------------------------------------------------
    diffusion_model = build_conditional_diffusion_model(IMG_SHAPE, NUM_CLASSES)

    # -----------------------------------------------------
    # Compile the model with optimizer and MSE loss
    # -----------------------------------------------------
    diffusion_model.compile(optimizer=optimizer, loss=mse_loss)

    # -----------------------------------------------------
    # SECTION 21A: Train or evaluate model based on the RUN_MODE flag
    # -----------------------------------------------------
    if RUN_MODE == "train":
        log_message("Starting Conditional Diffusion model training...")

        alpha_hat = create_alpha_schedule(timesteps=T)

        train_diffusion_model(
            model=diffusion_model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            alpha_hat=alpha_hat,
            epochs=EPOCHS,
            summary_writer=summary_writer,
            checkpoint_dir=CHECKPOINT_DIR,
            patience=PATIENCE

        )
        diffusion_model.save_weights(os.path.join(CHECKPOINT_DIR, "diffusion_final.h5"))

    elif RUN_MODE == "eval_only":
        log_message("Loading best saved diffusion model for evaluation...")
        diffusion_model.load_weights(os.path.join(CHECKPOINT_DIR, "diffusion_best.h5"))

    # -----------------------------------------------------
    # SECTION 21B: Generate synthetic samples per class using the trained model
    # -----------------------------------------------------
    generate_synthetic_per_class(
        model=diffusion_model,
        img_shape=IMG_SHAPE,
        num_classes=NUM_CLASSES,
        samples_per_class=5,
        timesteps=TIMESTEPS,
        alpha_hat=alpha_hat
    )

    # -----------------------------------------------------
    # SECTION 21C: Load synthetic samples from disk
    # -----------------------------------------------------
    x_synth, y_synth = load_synthetic_samples(
        root_dir=SYNTHETIC_DIR,
        num_classes=NUM_CLASSES,
        samples_per_class=100,
        img_shape=IMG_SHAPE
    )

    # =====================================
    # SECTION 21D: FID Evaluation (Diffusion Model)
    # =====================================
    log_message("Evaluating FID score for diffusion synthetic samples...")

    # Load real validation images (truncate to match synthetic sample size)
    x_real_fid = []
    for batch in val_dataset:
        x_real_fid.append(batch[0])
    x_real_fid = tf.concat(x_real_fid, axis=0)
    x_real_fid = tf.convert_to_tensor(x_real_fid[:x_synth.shape[0]])

    # Convert synthetic images to tensor
    x_fake_fid = tf.convert_to_tensor(x_synth)

    # Compute Fréchet Inception Distance (FID)
    fid_score_diff = calculate_fid(x_real_fid, x_fake_fid)
    log_message(f"FID Score (Conditional Diffusion Model): {fid_score_diff:.4f}")

    # Append FID result to comparison log file
    with open("diffusion_comparison_results.txt", "a") as f:
        f.write(f"FID Score (Diffusion): {fid_score_diff:.4f}\n")

    # Log FID score to TensorBoard
    with summary_writer.as_default():
        tf.summary.scalar("FID_Score_Diffusion", fid_score_diff, step=0)
        summary_writer.flush()

    # =====================================
    # SECTION 21E: Classifier Training and Evaluation
    # =====================================

    # Train classifier on real data only
    classifier_real, history_real = train_real_only_classifier(
        train_data=train_dataset,
        val_data=val_dataset,
        input_shape=IMG_SHAPE,
        num_classes=NUM_CLASSES,
        epochs=20,
        log_path="classifier_real_diffusion.h5"
    )

    # Convert real training dataset to NumPy arrays
    x_real, y_real = [], []
    for batch_x, batch_y in train_dataset:
        x_real.append(batch_x.numpy())
        y_real.append(batch_y.numpy())
    x_real = np.concatenate(x_real)
    y_real = np.concatenate(y_real)

    # Train classifier on combined real + synthetic data
    classifier_combined, history_combined = train_classifier_on_real_plus_synthetic(
        x_real, y_real,
        x_synth, y_synth,
        img_shape=IMG_SHAPE,
        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        epochs=20,
        log_dir="logs/classifier_combined",
        save_path="classifier_combined_diffusion.h5"
    )

    # Prepare test dataset for evaluation
    x_test = []
    y_test = []
    for batch_x, batch_y in test_dataset:
        x_test.append(batch_x.numpy())
        y_test.append(batch_y.numpy())

    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    # Evaluate both classifiers on the same test set
    results = evaluate_classifier_on_real_vs_combined(
        classifier_real, classifier_combined, x_test, y_test
    )

    # =====================================
    # SECTION 21F: Plot Graphs
    # =====================================

    # For Loss
    save_metric_plot(
        history=history_real,
        metrics=["loss", "val_loss"],
        title="Real-Only Classifier Loss (Diffusion)",
        ylabel="Loss",
        filename="real_only_loss_diffusion.png"
    )

    save_metric_plot(
        history=history_combined,
        metrics=["loss", "val_loss"],
        title="Real+Synthetic Classifier Loss (Diffusion)",
        ylabel="Loss",
        filename="combined_loss_diffusion.png"
    )

    # For Accuracy
    save_metric_plot(
        history=history_real,
        metrics=["accuracy", "val_accuracy"],
        title="Real-Only Classifier Accuracy (Diffusion)",
        ylabel="Accuracy",
        filename="real_only_accuracy_diffusion.png"
    )

    save_metric_plot(
        history=history_combined,
        metrics=["accuracy", "val_accuracy"],
        title="Real+Synthetic Classifier Accuracy (Diffusion)",
        ylabel="Accuracy",
        filename="combined_accuracy_diffusion.png"
    )

    # Log evaluation comparison results
    log_comparison_results(results)



