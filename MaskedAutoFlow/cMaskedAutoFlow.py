# =====================================
# SECTION 1: Imports and Initial Setup
# =====================================

import os
import yaml
import random
import numpy as np
from datetime import datetime
from keras.src.optimizers.adam import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras import layers, models, losses, optimizers, callbacks
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==============================
# Set Random Seeds
# ==============================
def set_random_seeds(seed=42):
    """Ensure reproducibility across NumPy, TensorFlow, and random."""
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
IMG_SHAPE = tuple(config["IMG_SHAPE"])        # (H, W, C)
BATCH_SIZE = config["BATCH_SIZE"]
EPOCHS = config["EPOCHS"]
LR = config["LR"]
NUM_CLASSES = config["NUM_CLASSES"]
RUN_MODE = config.get("mode", "train")
VERBOSE = config.get("verbose", True)
PATIENCE = config.get("patience", 10)
NUM_FLOW_LAYERS = config["model"]["num_flow_layers"]

# ==============================
# Define Output Paths
# ==============================
EXPERIMENT_NAME = "masked_auto_flow"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "USTC-TFC2016_malware")

LOG_FILE = os.path.join(BASE_DIR, f"{EXPERIMENT_NAME}_result.txt")
CHECKPOINT_DIR = os.path.join(BASE_DIR, f"{EXPERIMENT_NAME}_checkpoints")
SYNTHETIC_DIR = os.path.join(BASE_DIR, f"{EXPERIMENT_NAME}_synthetic_samples")
LOG_DIR = os.path.join(BASE_DIR, "logs", EXPERIMENT_NAME)

# ==============================
# Create Output Directories
# ==============================
for dir_path in [CHECKPOINT_DIR, SYNTHETIC_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ==============================
# Logging Utility
# ==============================
def log_message(message, display=True):
    """Print and log messages with timestamps to file and optionally to console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    if VERBOSE and display:
        print(full_msg)
    with open(LOG_FILE, "a") as f:
        f.write(full_msg + "\n")

def prepare_tf_dataset(x, y, batch_size=128, shuffle=True):
    """Converts NumPy arrays into a batched TensorFlow Dataset."""
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# =====================================
# SECTION 2A: Data Loading and Preprocessing
# =====================================

def load_malware_dataset(data_path, img_shape, num_classes, val_fraction=0.5):
    """
    Load and preprocess malware image data for Masked Auto Flow.

    Args:
        data_path (str): Path to directory containing .npy dataset files
        img_shape (tuple): Shape of each input image (H, W, C)
        num_classes (int): Number of class labels
        val_fraction (float): Fraction of test set to use as validation

    Returns:
        Tuple of: (x_train, y_train), (x_val, y_val), (x_test, y_test)
    """
    x_train = np.load(os.path.join(data_path, "train_data.npy"))
    y_train = np.load(os.path.join(data_path, "train_labels.npy"))
    x_test = np.load(os.path.join(data_path, "test_data.npy"))
    y_test = np.load(os.path.join(data_path, "test_labels.npy"))

    # Normalize inputs to [0, 1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # Flatten images if required by model later
    x_train = x_train.reshape((-1, np.prod(img_shape)))
    x_test = x_test.reshape((-1, np.prod(img_shape)))

    # One-hot encode labels
    y_train = np.eye(num_classes)[y_train]
    y_test = np.eye(num_classes)[y_test]

    # Split test set into validation and test sets
    split_idx = int(len(x_test) * val_fraction)
    x_val, y_val = x_test[:split_idx], y_test[:split_idx]
    x_test, y_test = x_test[split_idx:], y_test[split_idx:]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def get_data_loaders(x_train, y_train, x_val, y_val, x_test, y_test, batch_size):
    """
    Create PyTorch DataLoaders from numpy arrays.

    Args:
        x_train, y_train, x_val, y_val, x_test, y_test (np.ndarray): Preprocessed data
        batch_size (int): Batch size for loaders

    Returns:
        Tuple of train_loader, val_loader, test_loader
    """

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=1000).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    return train_ds, val_ds, test_ds



# =====================================
# SECTION 2B: Evaluation Metric Placeholders
# =====================================

# ------------------------------
# Metric 1: Fréchet Inception Distance (FID)
# ------------------------------
def calculate_fid(real_images, generated_images, model=None):
    """
    Compute Fréchet Inception Distance (FID) between real and generated images.

    Args:
        real_images (np.ndarray): Real samples [N, H, W, C]
        generated_images (np.ndarray): Generated samples [N, H, W, C]
        model: Preloaded InceptionV3 model or None

    Returns:
        float: FID score
    """
    import tensorflow as tf
    from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

    real_images = tf.image.resize(real_images, (299, 299))
    generated_images = tf.image.resize(generated_images, (299, 299))

    real_images = tf.image.grayscale_to_rgb(real_images)
    generated_images = tf.image.grayscale_to_rgb(generated_images)

    real_images = preprocess_input(real_images)
    generated_images = preprocess_input(generated_images)

    if model is None:
        model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    act1 = model.predict(real_images, verbose=0)
    act2 = model.predict(generated_images, verbose=0)

    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)


# ------------------------------
# Metric 2: Jensen-Shannon and KL Divergence
# ------------------------------
def js_divergence(p, q):
    """Compute Jensen-Shannon divergence between two distributions."""
    p, q = np.asarray(p, dtype=np.float32), np.asarray(q, dtype=np.float32)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def kl_divergence(p, q):
    """Compute Kullback-Leibler divergence between two distributions."""
    return entropy(p, q)


# ------------------------------
# Metric 3: Classifier Metrics
# ------------------------------
def evaluate_classifier(y_true, y_pred, average="macro"):
    """
    Evaluate classification performance.

    Args:
        y_true (np.ndarray): Ground truth labels
        y_pred (np.ndarray): Predicted labels

    Returns:
        dict: Dictionary of metrics
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average),
        "recall": recall_score(y_true, y_pred, average=average),
        "f1": f1_score(y_true, y_pred, average=average),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }


# ------------------------------
# Metric 4: Sample Diversity
# ------------------------------
def compute_sample_diversity(samples):
    """
    Estimate diversity score via feature-wise variance.

    Args:
        samples (np.ndarray): Sample matrix [N, D]

    Returns:
        float: Diversity score
    """
    return np.mean(np.var(samples, axis=0))



# =====================================
# SECTION 3: Masked Autoregressive Flow (MAF) Model Definition
# =====================================

# ==============================
# Masked Dense Layer
# ==============================
class MaskedDense(tf.keras.layers.Layer):
    """
    Fully connected dense layer with autoregressive masking.
    Implements MADE-style masks to preserve dependency structure.
    """

    def __init__(self, units, mask_type, input_dim):
        """
        Args:
            units (int): Number of output units
            mask_type (str): 'input', 'hidden', or 'output'
            input_dim (int): Dimensionality of input features
        """
        super(MaskedDense, self).__init__()
        self.units = units
        self.mask_type = mask_type
        self.input_dim = input_dim

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=True
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True
        )
        self.mask = self.create_mask(input_shape[-1], self.units)

    def call(self, inputs):
        masked_kernel = self.kernel * self.mask
        return tf.matmul(inputs, masked_kernel) + self.bias

    def create_mask(self, in_features, out_features):
        """
        Create autoregressive mask matrix based on degrees.

        Returns:
            Tensor: Binary mask of shape [in_features, out_features]
        """
        input_degrees = tf.range(1, in_features + 1, dtype=tf.int32)
        if self.mask_type == "input":
            output_degrees = tf.range(1, out_features + 1, dtype=tf.int32)
        elif self.mask_type == "hidden":
            output_degrees = tf.random.uniform(
                shape=(out_features,), minval=1, maxval=in_features + 1, dtype=tf.int32)
        else:  # "output"
            output_degrees = tf.range(1, out_features + 1, dtype=tf.int32)

        mask = tf.cast(
            tf.expand_dims(input_degrees, -1) <= tf.expand_dims(output_degrees, 0),
            tf.float32
        )
        return mask


# ==============================
# MADE Subnetwork for One Flow Step
# ==============================
class MADE(tf.keras.Model):
    """
    Masked Autoencoder for Distribution Estimation (MADE).
    Forms the building block for each MAF transformation.
    """

    def __init__(self, input_dim, hidden_dims=[128, 128]):
        """
        Args:
            input_dim (int): Input dimensionality
            hidden_dims (list): List of hidden layer sizes
        """
        super(MADE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.net = []

        prev_dim = input_dim
        for h in hidden_dims:
            self.net.append(MaskedDense(h, mask_type="hidden", input_dim=prev_dim))
            self.net.append(tf.keras.layers.ReLU())
            prev_dim = h

        self.mu_layer = MaskedDense(input_dim, mask_type="output", input_dim=prev_dim)
        self.log_sigma_layer = MaskedDense(input_dim, mask_type="output", input_dim=prev_dim)

    def call(self, x):
        """
        Forward pass through MADE network.

        Args:
            x (Tensor): Input tensor [batch_size, input_dim]

        Returns:
            Tuple: Mean and log-variance tensors [batch_size, input_dim]
        """
        h = x
        for layer in self.net:
            h = layer(h)
        mu = self.mu_layer(h)
        log_sigma = self.log_sigma_layer(h)
        return mu, log_sigma


# ==============================
# Full MAF Model Definition
# ==============================
class MAF(tf.keras.Model):
    """
    Masked Autoregressive Flow (MAF) model.
    Consists of stacked MADE blocks for expressive density estimation.
    """

    def __init__(self, input_dim, num_flows=5, hidden_dims=[128, 128]):
        super(MAF, self).__init__()
        self.input_dim = input_dim
        self.flows = [MADE(input_dim, hidden_dims) for _ in range(num_flows)]

    def build(self, input_shape):
        # Create variables by calling the model once
        dummy_input = tf.zeros((1, self.input_dim), dtype=tf.float32)
        self.call(dummy_input)

    def call(self, x):
        """
        Forward pass for Keras compatibility (used by model()).
        Returns only z for building and tracing.
        """
        z, _ = self.forward(x)

        return z

    def forward(self, x):
        """
        Actual forward logic with log_det_jacobian for log_prob calculation.
        """
        log_det_jacobian = 0.0
        z = x
        for flow in self.flows:
            mu, log_sigma = flow(z)
            z = (z - mu) * tf.exp(-log_sigma)
            log_det_jacobian += -tf.reduce_sum(log_sigma, axis=1)

        return z, log_det_jacobian

    def log_prob(self, x):
        """
        Compute log-likelihood under base Gaussian with flow transformation.
        """
        z, log_det = self.forward(x)
        log_base = -0.5 * tf.reduce_sum(z ** 2 + tf.math.log(2.0 * np.pi), axis=1)

        return log_base + log_det


def build_maf_model(input_dim, num_layers=5, hidden_dims=[128, 128]):
    return MAF(input_dim=input_dim, num_flows=num_layers, hidden_dims=hidden_dims)


# =====================================
# SECTION 4: MAF Training Loop with Early Stopping
# =====================================

def train_maf_model(model, train_data, val_data, config, writer, checkpoint_dir):
    """
    Train the Masked Autoregressive Flow (MAF) model.

    Args:
        model (tf.keras.Model): Initialized MAF model
        train_data (tf.data.Dataset): Training data (batched)
        val_data (tf.data.Dataset): Validation data (batched)
        config (dict): Training configurations from YAML
        writer (tf.summary.SummaryWriter): TensorBoard summary writer
        save_dir (str): Directory to store best model checkpoint

    Returns:
        tf.keras.Model: Best-performing MAF model (on validation loss)
    """

    optimizer = tf.keras.optimizers.Adam(learning_rate=config["LR"])
    epochs = config["EPOCHS"]
    early_stop_patience = config.get("PATIENCE", 10)
    clip_grad = config.get("CLIP_GRAD", 1.0)

    best_val_loss = float("inf")
    patience_counter = 0

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss_avg = tf.keras.metrics.Mean()
        val_loss_avg = tf.keras.metrics.Mean()

        input_dim = np.prod(IMG_SHAPE)
        dummy_input = tf.zeros((1, input_dim), dtype=tf.float32)
        model(dummy_input)  # builds the model explicitly

        # Training
        for x_batch, _ in train_data:
            with tf.GradientTape() as tape:
                neg_log_likelihood = -tf.reduce_mean(model.log_prob(x_batch))
            grads = tape.gradient(neg_log_likelihood, model.trainable_variables)
            grads, _ = tf.clip_by_global_norm(grads, clip_grad)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_loss_avg.update_state(neg_log_likelihood)

        # Validation
        for x_batch, _ in val_data:
            val_nll = -tf.reduce_mean(model.log_prob(x_batch))
            val_loss_avg.update_state(val_nll)

        epoch_train_loss = train_loss_avg.result().numpy()
        epoch_val_loss = val_loss_avg.result().numpy()

        # Logging
        with writer.as_default():
            tf.summary.scalar("Loss/Train_NLL", epoch_train_loss, step=epoch)
            tf.summary.scalar("Loss/Val_NLL", epoch_val_loss, step=epoch)

        print(
            f"Epoch {epoch:03d} | Train NLL: {epoch_train_loss:.4f} | Val NLL: {epoch_val_loss:.4f} | Time: {time.time() - start_time:.2f}s")

        # Early Stopping and Saving
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0


            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(checkpoint_dir, f"maf_best_{timestamp}.h5")
            print(f"Saving best model weights to: {model_path}")

            # Ensure any stale files are removed before saving
            if os.path.exists(model_path):
                os.remove(model_path)
                time.sleep(1)  # ensure OS file system flushes removal

            model.save_weights(model_path, overwrite=True, save_format='h5')

        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{early_stop_patience}")
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break

    return model


# =====================================
# SECTION 5: Synthetic Sampling and Saving Outputs
# =====================================

def generate_maf_samples(model, num_samples, input_dim, img_shape):
    """
    Generate synthetic samples from a trained MAF model.

    Args:
        model (tf.keras.Model): Trained MAF model
        num_samples (int): Number of samples to generate
        input_dim (int): Dimensionality of each sample
        img_shape (tuple): Shape of each image (H, W, C)

    Returns:
        ndarray: Synthetic samples in image format [N, H, W, C]
    """
    # Sample latent vectors from standard normal
    z = tf.random.normal(shape=(num_samples, input_dim))
    x_samples = model.inverse(z).numpy()
    x_samples = np.clip(x_samples, 0.0, 1.0)  # Ensure in [0, 1]

    # Reshape to image format
    return x_samples.reshape((-1, *img_shape))


def save_synthetic_samples(samples, output_dir, prefix="maf"):
    """
    Save each synthetic image sample to a .npy file.

    Args:
        samples (ndarray): Samples with shape [N, H, W, C]
        output_dir (str): Directory to save the .npy files
        prefix (str): Prefix for filenames
    """
    os.makedirs(output_dir, exist_ok=True)
    for idx, sample in enumerate(samples):
        np.save(os.path.join(output_dir, f"{prefix}_{idx}.npy"), sample)


def visualize_samples(samples, num_display=9):
    """
    Display a grid of generated image samples.

    Args:
        samples (ndarray): Image samples [N, H, W, C]
        num_display (int): Number of images to show (must be square number)
    """
    grid_size = int(np.sqrt(num_display))
    plt.figure(figsize=(grid_size * 2, grid_size * 2))

    for i in range(num_display):
        plt.subplot(grid_size, grid_size, i + 1)
        img = samples[i]
        if img.shape[-1] == 1:
            plt.imshow(img.squeeze(), cmap="gray")
        else:
            plt.imshow(img)
        plt.axis("off")

    plt.tight_layout()
    plt.show()



# =====================================
# SECTION 6: Evaluation Using Synthetic Samples
# =====================================

def calculate_fid(real_images, generated_images):
    """
    Compute Fréchet Inception Distance (FID) between real and generated samples.

    Args:
        real_images (ndarray): Real images in [0,1], shape [N, H, W, C]
        generated_images (ndarray): Generated images in [0,1], shape [N, H, W, C]

    Returns:
        float: FID score
    """
    # Resize to (299, 299, 3)
    real = tf.image.resize(real_images, (299, 299))
    fake = tf.image.resize(generated_images, (299, 299))

    real = tf.image.grayscale_to_rgb(real)
    fake = tf.image.grayscale_to_rgb(fake)

    real = preprocess_input(real)
    fake = preprocess_input(fake)

    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    act1 = model.predict(real, verbose=0)
    act2 = model.predict(fake, verbose=0)

    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def js_divergence(p, q):
    """Compute Jensen-Shannon divergence between two distributions."""
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def kl_divergence(p, q):
    """Compute KL divergence between two distributions."""
    return entropy(p, q)


def compute_sample_diversity(samples):
    """
    Compute diversity score using feature variance.

    Args:
        samples (ndarray): Flattened generated samples [N, D]

    Returns:
        float: Mean feature-wise variance
    """
    return float(np.mean(np.var(samples.reshape(samples.shape[0], -1), axis=0)))


def evaluate_generation_quality(real_images, generated_images, alpha=1e-6):
    """
    Full evaluation pipeline for comparing generated and real images.

    Args:
        real_images (ndarray): Real images in [0,1], shape [N, H, W, C]
        generated_images (ndarray): Generated images in [0,1], shape [N, H, W, C]
        alpha (float): Smoothing factor for histogram-based divergence metrics

    Returns:
        dict: Dictionary containing FID, JS, KL, and diversity score
    """
    # -----------------------------
    # Metric 1: FID Score
    # -----------------------------
    fid = calculate_fid(real_images, generated_images)

    # -----------------------------
    # Metric 2: Histogram Divergences (Pixel-wise)
    # -----------------------------
    def get_histogram(imgs):
        return np.histogram(imgs.flatten(), bins=256, range=(0, 1), density=True)[0] + alpha

    p_real = get_histogram(real_images)
    p_fake = get_histogram(generated_images)

    js = js_divergence(p_real, p_fake)
    kl = kl_divergence(p_real, p_fake)

    # -----------------------------
    # Metric 3: Diversity Score
    # -----------------------------
    diversity = compute_sample_diversity(generated_images)

    return {
        "FID": fid,
        "JS_Divergence": js,
        "KL_Divergence": kl,
        "Diversity_Score": diversity
    }



# =====================================
# SECTION 7: Classifier Utility Evaluation
# =====================================

def build_classifier(input_shape, num_classes):
    """
    Build a simple CNN classifier.

    Args:
        input_shape (tuple): Input image shape (H, W, C)
        num_classes (int): Number of output classes

    Returns:
        model: Compiled Keras CNN model
    """
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=optimizers.Adam(),
                  loss=losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


def train_classifier(model, x_train, y_train, x_val, y_val, log_dir, epochs=20):
    """
    Train classifier with early stopping and TensorBoard.

    Args:
        model: Compiled Keras model
        x_train, y_train: Training set (one-hot encoded)
        x_val, y_val: Validation set
        log_dir (str): TensorBoard logging directory
        epochs (int): Max epochs

    Returns:
        model: Trained model
    """
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    tensorboard_cb = callbacks.TensorBoard(log_dir=log_dir)

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=64,
        callbacks=[early_stop, tensorboard_cb],
        verbose=1
    )
    return model


def evaluate_classifier(model, x_test, y_test):
    """
    Evaluate a trained classifier using full metrics.

    Args:
        model: Trained Keras model
        x_test, y_test: Test data (one-hot)

    Returns:
        dict: Accuracy, Precision, Recall, F1, Confusion Matrix
    """
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='macro'),
        "recall": recall_score(y_true, y_pred, average='macro'),
        "f1": f1_score(y_true, y_pred, average='macro'),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }



# =====================================
# SECTION 8: Main Runner Script
# =====================================

if __name__ == "__main__":

    # -----------------------------
    # Select Device
    # -----------------------------
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        device = "GPU"
    else:
        device = "CPU"
    log_message(f"Using device: {device}")

    # -----------------------------
    # Load and Prepare Dataset
    # -----------------------------
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_malware_dataset(
        data_path=DATA_PATH,
        img_shape=IMG_SHAPE,
        num_classes=NUM_CLASSES,
        val_fraction=0.5
    )

    # -----------------------------
    # Convert Numpy Arrays to TF Datasets
    # -----------------------------
    train_ds = prepare_tf_dataset(x_train, y_train, batch_size=BATCH_SIZE)
    val_ds = prepare_tf_dataset(x_val, y_val, batch_size=BATCH_SIZE)
    test_ds = prepare_tf_dataset(x_test, y_test, batch_size=BATCH_SIZE)

    # -----------------------------
    # Build MAF Model
    # -----------------------------
    maf_model = build_maf_model(input_dim=np.prod(IMG_SHAPE), num_layers=NUM_FLOW_LAYERS)
    maf_model.build((None, np.prod(IMG_SHAPE)))  # ensure variables initialized

    # -----------------------------
    # Train MAF Model with Early Stopping
    # -----------------------------
    writer = tf.summary.create_file_writer(os.path.join(LOG_DIR, "maf_tensorboard"))

    train_maf_model(
        model=maf_model,
        train_data=train_ds,
        val_data=val_ds,
        config=config,
        writer=writer,
        checkpoint_dir=CHECKPOINT_DIR
    )

    # -----------------------------
    # Generate and Save Synthetic Samples
    # -----------------------------
    synthetic_samples = sample_from_maf(
        maf_model,
        num_samples=1000,
        input_dim=np.prod(IMG_SHAPE)
    )
    save_synthetic_samples(synthetic_samples, save_dir=SYNTHETIC_DIR, prefix="maf")
    visualize_samples(synthetic_samples, IMG_SHAPE, num_display=16)

    # -----------------------------
    # Evaluate Generated Samples
    # -----------------------------
    eval_results = evaluate_generated_samples(
        real_data=x_val[:1000].reshape(-1, *IMG_SHAPE),
        generated_data=synthetic_samples.reshape(-1, *IMG_SHAPE),
        img_shape=IMG_SHAPE
    )
    log_message(f"MAF Evaluation Metrics: {eval_results}")

    # -----------------------------
    # Classifier Utility Evaluation
    # -----------------------------

    # (1) Train Classifier on Real Data Only
    clf_real = build_classifier(input_shape=IMG_SHAPE, num_classes=NUM_CLASSES)
    train_ds_classifier = prepare_tf_dataset(x_train, y_train, batch_size=BATCH_SIZE)
    val_ds_classifier = prepare_tf_dataset(x_val, y_val, batch_size=BATCH_SIZE)
    test_ds_classifier = prepare_tf_dataset(x_test, y_test, batch_size=BATCH_SIZE)

    clf_real = train_classifier(clf_real, train_ds_classifier, val_ds_classifier, epochs=20)
    real_metrics = evaluate_classifier_utility(clf_real, test_ds_classifier)
    log_message(f"[Classifier (Real Only)] {real_metrics}")

    # (2) Train Classifier on Real + Synthetic Data
    y_synth = y_train[:len(synthetic_samples)]  # use aligned labels for synthetic samples
    x_combined = np.concatenate([x_train, synthetic_samples], axis=0)
    y_combined = np.concatenate([y_train, y_synth], axis=0)

    combined_ds_classifier = prepare_tf_dataset(x_combined, y_combined, batch_size=BATCH_SIZE)

    clf_combined = build_classifier(input_shape=IMG_SHAPE, num_classes=NUM_CLASSES)
    clf_combined = train_classifier(clf_combined, combined_ds_classifier, val_ds_classifier, epochs=20)
    combined_metrics = evaluate_classifier_utility(clf_combined, test_ds_classifier)
    log_message(f"[Classifier (Real + Synthetic)] {combined_metrics}")

    # -----------------------------
    # Save Final Accuracy Comparison
    # -----------------------------
    comparison_results = {
        "real_only": real_metrics["accuracy"],
        "real_plus_synthetic": combined_metrics["accuracy"]
    }

    log_comparison_results(
        comparison_results,
        output_path=os.path.join(BASE_DIR, "maf_classifier_comparison.txt")
    )
