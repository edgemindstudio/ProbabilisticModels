# =====================================
# SECTION 1: Imports and Initial Setup
# =====================================

import os
import time
import yaml
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from scipy.linalg import sqrtm
from tensorflow.keras import layers, models

# =====================================
# Set Random Seeds
# =====================================
def set_random_seeds(seed=42):
    """Ensure reproducibility across TensorFlow, NumPy, and random."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_random_seeds()

# =====================================
# Load Configuration File
# =====================================
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# =====================================
# Load Hyperparameters from Config
# =====================================
IMG_SHAPE = tuple(config["IMG_SHAPE"])        # (H, W, C)
VISIBLE_UNITS = np.prod(IMG_SHAPE)            # Flattened image size
HIDDEN_UNITS = config.get("HIDDEN_UNITS", 256)
BATCH_SIZE = config["BATCH_SIZE"]
EPOCHS = config["EPOCHS"]
LR = config["LR"]
NUM_CLASSES = config["NUM_CLASSES"]
RUN_MODE = config.get("mode", "train")
VERBOSE = config.get("verbose", True)
K_STEPS = config.get("CD_K", 1)

# =====================================
# Define Output Paths
# =====================================
EXPERIMENT_NAME = "rbm_tf_experiment"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "USTC-TFC2016_malware")

LOG_FILE = os.path.join(BASE_DIR, f"{EXPERIMENT_NAME}_result.txt")
CHECKPOINT_DIR = os.path.join(BASE_DIR, f"{EXPERIMENT_NAME}_checkpoints")
SYNTHETIC_DIR = os.path.join(BASE_DIR, f"{EXPERIMENT_NAME}_synthetic_samples")
LOG_DIR = os.path.join(BASE_DIR, "logs", EXPERIMENT_NAME)

# =====================================
# Create Output Directories
# =====================================
for dir_path in [CHECKPOINT_DIR, SYNTHETIC_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# =====================================
# Logging Utility
# =====================================
def log_message(message, display=True):
    """Print and log messages with timestamps to file and optionally to console."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    if VERBOSE and display:
        print(full_msg)
    with open(LOG_FILE, "a") as f:
        f.write(full_msg + "\n")


def log_comparison_results(results_dict, output_path):
    with open(output_path, "w") as f:
        f.write("Classifier Accuracy Comparison:\n")
        for key, value in results_dict.items():
            f.write(f"{key}: {value:.4f}\n")



# =====================================
# SECTION 2A: Data Loading and Preprocessing
# =====================================

def load_malware_dataset(data_path, img_shape, num_classes, val_fraction=0.5):
    """
    Load malware traffic images and preprocess for RBM training.

    Steps:
    - Normalize to [0, 1]
    - Binarize images
    - Flatten to vectors
    - One-hot encode labels
    - Split test set into validation and test sets

    Args:
        data_path (str): Path to dataset folder
        img_shape (tuple): (H, W, C)
        num_classes (int): Total number of classes
        val_fraction (float): Ratio of validation data from test set

    Returns:
        Tuple of train, val, and test sets: (x_train, y_train), (x_val, y_val), (x_test, y_test)
    """
    # Load data
    x_train = np.load(os.path.join(data_path, "train_data.npy"))
    y_train = np.load(os.path.join(data_path, "train_labels.npy"))
    x_test = np.load(os.path.join(data_path, "test_data.npy"))
    y_test = np.load(os.path.join(data_path, "test_labels.npy"))

    # Normalize to [0, 1] and binarize
    x_train = (x_train.astype(np.float32) / 255.0 > 0.5).astype(np.float32)
    x_test = (x_test.astype(np.float32) / 255.0 > 0.5).astype(np.float32)

    # Flatten images
    x_train = x_train.reshape((-1, np.prod(img_shape)))
    x_test = x_test.reshape((-1, np.prod(img_shape)))

    # One-hot encode labels
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

    # Split test into val and test
    split_idx = int(len(x_test) * val_fraction)
    x_val, y_val = x_test[:split_idx], y_test[:split_idx]
    x_test, y_test = x_test[split_idx:], y_test[split_idx:]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def create_tf_dataloaders(x_train, y_train, x_val, y_val, x_test, y_test, batch_size):
    """
    Convert arrays into TensorFlow tf.data.Dataset pipelines.

    Args:
        x_train, y_train: Training set
        x_val, y_val: Validation set
        x_test, y_test: Test set
        batch_size (int): Batch size for training and evaluation

    Returns:
        train_ds, val_ds, test_ds: Batched TensorFlow datasets
    """
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=10000).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    return train_ds, val_ds, test_ds



# =====================================
# SECTION 2B: Evaluation Metric Placeholders
# =====================================

# -------------------------------
# Metric 1: Fréchet Inception Distance (FID)
# -------------------------------
def calculate_fid(real_images, generated_images):
    """
    Compute Fréchet Inception Distance (FID) using InceptionV3.

    Args:
        real_images (ndarray): Real images [N, H, W, 1]
        generated_images (ndarray): Generated images [N, H, W, 1]

    Returns:
        float: FID score
    """

    # Resize to 299x299 and convert to RGB
    real_images = tf.image.resize(real_images, (299, 299))
    fake_images = tf.image.resize(generated_images, (299, 299))
    real_images = tf.image.grayscale_to_rgb(real_images)
    fake_images = tf.image.grayscale_to_rgb(fake_images)

    real_images = preprocess_input(real_images)
    fake_images = preprocess_input(fake_images)

    # Load InceptionV3 model
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    act1 = model.predict(real_images, verbose=0)
    act2 = model.predict(fake_images, verbose=0)

    # Calculate mean and covariance
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2)
    covmean = sqrtm(sigma1 @ sigma2)

    # Handle imaginary values from sqrtm
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


# -------------------------------
# Metric 2: JS Divergence and KL Divergence
# -------------------------------
def js_divergence(p, q):
    """Jensen-Shannon divergence between two distributions."""
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def kl_divergence(p, q):
    """KL divergence between two distributions."""
    return entropy(p, q)


# -------------------------------
# Metric 3: Classifier Evaluation
# -------------------------------
def evaluate_classifier(y_true, y_pred, average="macro"):
    """
    Compute evaluation metrics for classifier performance.

    Args:
        y_true (list or array): True labels
        y_pred (list or array): Predicted labels
        average (str): Averaging strategy for precision/recall/f1

    Returns:
        dict: Accuracy, precision, recall, f1, confusion_matrix
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average),
        "recall": recall_score(y_true, y_pred, average=average),
        "f1": f1_score(y_true, y_pred, average=average),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }


# -------------------------------
# Metric 4: Diversity Score
# -------------------------------
def compute_sample_diversity(samples):
    """
    Compute sample diversity as variance across features.

    Args:
        samples (ndarray): [N, D] synthetic samples

    Returns:
        float: Diversity score (higher is more diverse)
    """
    return np.mean(np.var(samples, axis=0))



# =====================================
# SECTION 3: RBM Model Definition (TensorFlow Version)
# =====================================

class RBM(tf.keras.Model):
    """
    Restricted Boltzmann Machine implemented in TensorFlow.
    Trained using Contrastive Divergence (CD-k).
    """

    def __init__(self, visible_units, hidden_units):
        """
        Initialize RBM weights and biases.

        Args:
            visible_units (int): Size of visible/input layer
            hidden_units (int): Size of hidden layer
        """
        super(RBM, self).__init__()
        self.visible_units = visible_units
        self.hidden_units = hidden_units

        # Initialize weights and biases
        initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
        self.W = tf.Variable(initializer(shape=(visible_units, hidden_units)), trainable=True)
        self.h_bias = tf.Variable(tf.zeros([hidden_units]), trainable=True)
        self.v_bias = tf.Variable(tf.zeros([visible_units]), trainable=True)

    def sample_prob(self, probs):
        """Sample from Bernoulli distribution."""
        return tf.nn.relu(tf.sign(probs - tf.random.uniform(tf.shape(probs))))

    def sample_h_given_v(self, v):
        """Sample hidden units given visible units."""
        h_prob = tf.nn.sigmoid(tf.matmul(v, self.W) + self.h_bias)
        return self.sample_prob(h_prob), h_prob

    def sample_v_given_h(self, h):
        """Sample visible units given hidden units."""
        v_prob = tf.nn.sigmoid(tf.matmul(h, tf.transpose(self.W)) + self.v_bias)
        return self.sample_prob(v_prob), v_prob

    def gibbs_sampling(self, v0, k=1):
        """
        Perform k steps of Gibbs sampling starting from v0.

        Args:
            v0 (tensor): Initial visible units
            k (int): Number of CD steps

        Returns:
            vk: Sampled visible units after k steps
        """
        vk = v0
        for _ in range(k):
            hk, _ = self.sample_h_given_v(vk)
            vk, _ = self.sample_v_given_h(hk)
        return vk

    def call(self, inputs):
        """
        Forward reconstruction: v → h → v̂.

        Args:
            inputs (tensor): Visible inputs

        Returns:
            tensor: Reconstructed inputs
        """
        h_sample, _ = self.sample_h_given_v(inputs)
        v_recon, _ = self.sample_v_given_h(h_sample)
        return v_recon

    def free_energy(self, v):
        """
        Compute free energy for a batch of visible vectors.

        Args:
            v (tensor): Visible input vectors

        Returns:
            tensor: Free energy values
        """
        vbias_term = tf.reduce_sum(v * self.v_bias, axis=1)
        wx_b = tf.matmul(v, self.W) + self.h_bias
        hidden_term = tf.reduce_sum(tf.math.softplus(wx_b), axis=1)

        return -vbias_term - hidden_term



# =====================================
# SECTION 4: RBM Training Loop and Utilities (TensorFlow Version)
# =====================================

def train_one_epoch(rbm, train_dataset, optimizer, k_steps):
    """
    Train RBM for one epoch using CD-k (Contrastive Divergence).

    Args:
        rbm (RBM): RBM model instance
        train_dataset (tf.data.Dataset): Batches of training data
        optimizer (tf.keras.optimizers.Optimizer): Optimizer instance
        k_steps (int): Number of Gibbs sampling steps

    Returns:
        float: Average reconstruction loss for the epoch
    """
    total_loss = 0.0
    num_batches = 0

    for v0, _ in train_dataset:
        with tf.GradientTape() as tape:
            # Gibbs sampling
            vk = rbm.gibbs_sampling(v0, k=k_steps)

            # Reconstruction loss (mean squared error)
            loss = tf.reduce_mean(tf.square(v0 - vk))

        grads = tape.gradient(loss, rbm.trainable_variables)
        optimizer.apply_gradients(zip(grads, rbm.trainable_variables))

        total_loss += loss.numpy()
        num_batches += 1

    return total_loss / num_batches


def validate_rbm(rbm, val_dataset):
    """
    Evaluate RBM on validation dataset using reconstruction loss.

    Args:
        rbm (RBM): Trained RBM model
        val_dataset (tf.data.Dataset): Validation data

    Returns:
        float: Average reconstruction loss
    """
    total_loss = 0.0
    num_batches = 0

    for v, _ in val_dataset:
        v_recon = rbm(v)
        loss = tf.reduce_mean(tf.square(v - v_recon))
        total_loss += loss.numpy()
        num_batches += 1

    return total_loss / num_batches


def train_rbm(
    rbm, train_dataset, val_dataset, optimizer, k_steps, epochs,
    writer, checkpoint_path, patience=10
):
    """
    Full training loop for RBM with early stopping and TensorBoard logging.

    Args:
        rbm (RBM): RBM model
        train_dataset (tf.data.Dataset): Training data
        val_dataset (tf.data.Dataset): Validation data
        optimizer (tf.keras.optimizers.Optimizer): Optimizer
        k_steps (int): CD-k Gibbs sampling steps
        epochs (int): Maximum number of epochs
        writer (tf.summary.SummaryWriter): TensorBoard writer
        checkpoint_path (str): Path to save best model
        patience (int): Early stopping patience
    """
    log_message("Starting RBM training (TensorFlow)...")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        start = time.time()

        train_loss = train_one_epoch(rbm, train_dataset, optimizer, k_steps)
        val_loss = validate_rbm(rbm, val_dataset)
        epoch_time = time.time() - start

        # Logging
        log_message(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Time = {epoch_time:.2f}s")
        writer.add_scalar("Loss/Train", train_loss, step=epoch)
        writer.add_scalar("Loss/Validation", val_loss, step=epoch)
        writer.add_scalar("Time/EpochSeconds", epoch_time, step=epoch)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            rbm.save_weights(checkpoint_path)
            log_message("New best model saved.")
        else:
            patience_counter += 1
            log_message(f"Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                log_message("Early stopping triggered.")
                break



# =====================================
# SECTION 5: Synthetic Sampling and Saving Outputs
# =====================================

def sample_synthetic(rbm, num_samples, k_steps, visible_units):
    """
    Generate synthetic samples using Gibbs sampling from the trained RBM.

    Args:
        rbm (RBM): Trained RBM model
        num_samples (int): Number of samples to generate
        k_steps (int): Number of Gibbs sampling steps
        visible_units (int): Size of input vector (flattened image)

    Returns:
        ndarray: Binary synthetic samples of shape [num_samples, visible_units]
    """
    # Start with random binary visible vectors
    v = tf.cast(tf.random.uniform([num_samples, visible_units]) > 0.5, tf.float32)

    for _ in range(k_steps):
        # Hidden activation from visible input
        h_prob = tf.sigmoid(tf.matmul(v, rbm.W) + rbm.h_bias)
        h = tf.cast(tf.random.uniform(tf.shape(h_prob)) < h_prob, tf.float32)

        # Visible reconstruction from hidden sample
        v_prob = tf.sigmoid(tf.matmul(h, tf.transpose(rbm.W)) + rbm.v_bias)
        v = tf.cast(tf.random.uniform(tf.shape(v_prob)) < v_prob, tf.float32)

    return v.numpy()


def save_synthetic_samples(samples, save_dir, prefix="sample"):
    """
    Save synthetic samples as individual `.npy` files.

    Args:
        samples (ndarray): Synthetic binary samples [N, D]
        save_dir (str): Output directory
        prefix (str): Filename prefix
    """
    os.makedirs(save_dir, exist_ok=True)
    for idx, samp in enumerate(samples):
        np.save(os.path.join(save_dir, f"{prefix}_{idx}.npy"), samp)


def visualize_samples(samples, img_shape, num_display=9):
    """
    Plot a grid of synthetic samples for visual inspection.

    Args:
        samples (ndarray): Synthetic binary samples [N, D]
        img_shape (tuple): Original image shape (H, W, C)
        num_display (int): Number of samples to show (must be square number)
    """
    grid_size = int(np.sqrt(num_display))
    plt.figure(figsize=(grid_size * 2, grid_size * 2))

    for i in range(num_display):
        plt.subplot(grid_size, grid_size, i + 1)
        img = samples[i].reshape(img_shape)
        plt.imshow(img.squeeze(), cmap="gray")
        plt.axis("off")

    plt.tight_layout()
    plt.show()



# =====================================
# SECTION 6: Evaluation Using Synthetic Samples
# =====================================

def evaluate_rbm_generation(rbm, x_val, img_shape, num_samples, k_steps, alpha=1e-6):
    """
    Evaluate the quality of synthetic samples from RBM using multiple metrics.

    Args:
        rbm (RBM): Trained RBM model
        x_val (ndarray): Validation data from real dataset [N, D]
        img_shape (tuple): Image shape for reshaping (H, W, C)
        num_samples (int): Number of synthetic samples to generate
        k_steps (int): Number of Gibbs steps in sampling
        alpha (float): Smoothing factor for histograms

    Returns:
        dict: Dictionary with FID, JS, KL, and diversity metrics
    """
    log_message("Evaluating RBM synthetic sample quality...")

    # Generate synthetic samples
    synth = sample_synthetic(rbm, num_samples=num_samples, k_steps=k_steps, visible_units=np.prod(img_shape))

    # Reshape to image format
    real_images = x_val[:num_samples].reshape(-1, *img_shape)
    fake_images = synth.reshape(-1, *img_shape)

    # ----------------------
    # Metric 1: FID Score
    # ----------------------
    fid_score = calculate_fid(real_images, fake_images)
    log_message(f"FID Score (RBM): {fid_score:.4f}")

    # ----------------------
    # Metric 2: Pixel Histograms → JS/KL Divergence
    # ----------------------
    def get_pixel_distribution(images):
        hist = np.histogram(images.flatten(), bins=256, range=(0, 1), density=True)[0]
        return hist + alpha  # Add smoothing

    p_real = get_pixel_distribution(real_images)
    p_fake = get_pixel_distribution(fake_images)

    js = js_divergence(p_real, p_fake)
    kl = kl_divergence(p_real, p_fake)
    log_message(f"JS Divergence: {js:.6f}, KL Divergence: {kl:.6f}")

    # ----------------------
    # Metric 3: Sample Diversity Score
    # ----------------------
    diversity = compute_sample_diversity(synth)
    log_message(f"Diversity Score: {diversity:.6f}")

    return {
        "FID": fid_score,
        "JS": js,
        "KL": kl,
        "Diversity": diversity
    }




# =====================================
# SECTION 7: Classifier Utility Evaluation
# =====================================

def build_classifier(input_shape, num_classes):
    """
    Build a simple CNN classifier using Keras Sequential API.

    Args:
        input_shape (tuple): Image shape (H, W, C)
        num_classes (int): Total output classes

    Returns:
        model (tf.keras.Model): Compiled classifier model
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def prepare_tf_dataset(x, y, batch_size=64):
    """
    Create TensorFlow batched datasets from numpy arrays.

    Args:
        x (ndarray): Input features
        y (ndarray): One-hot labels
        batch_size (int): Batch size

    Returns:
        tf.data.Dataset: Batched dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def train_classifier(model, train_ds, val_ds, epochs, log_dir):
    """
    Train classifier with early stopping and TensorBoard logging.

    Args:
        model (tf.keras.Model): Compiled classifier
        train_ds (tf.data.Dataset): Training dataset
        val_ds (tf.data.Dataset): Validation dataset
        epochs (int): Maximum training epochs
        log_dir (str): Directory for TensorBoard logs

    Returns:
        model (tf.keras.Model): Trained model
    """
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stop, tensorboard_cb],
        verbose=2
    )

    return model


def evaluate_classifier_utility(model, test_ds):
    """
    Evaluate classifier performance on test data.

    Args:
        model (tf.keras.Model): Trained model
        test_ds (tf.data.Dataset): Batched test dataset

    Returns:
        dict: Accuracy, precision, recall, F1, confusion matrix
    """
    from sklearn.metrics import classification_report, confusion_matrix

    y_true = []
    y_pred = []

    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch)
        y_true.extend(tf.argmax(y_batch, axis=1).numpy())
        y_pred.extend(tf.argmax(preds, axis=1).numpy())

    report = classification_report(y_true, y_pred, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": report["accuracy"],
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1_score": report["macro avg"]["f1-score"],
        "confusion_matrix": matrix
    }

    log_message(f"[Classifier] Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")

    return metrics



# =====================================
# SECTION 8: Main Runner Script
# =====================================

if __name__ == "__main__":
    import tensorflow as tf

    log_message("Starting RBM Experiment (TensorFlow Version)")

    # -----------------------------
    # Load Dataset
    # -----------------------------
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_malware_dataset(
        DATA_PATH, IMG_SHAPE, NUM_CLASSES, val_fraction=0.5
    )

    # -----------------------------
    # Create TF Datasets
    # -----------------------------
    train_ds = prepare_tf_dataset(x_train.reshape(-1, *IMG_SHAPE), y_train, BATCH_SIZE)
    val_ds = prepare_tf_dataset(x_val.reshape(-1, *IMG_SHAPE), y_val, BATCH_SIZE)
    test_ds = prepare_tf_dataset(x_test.reshape(-1, *IMG_SHAPE), y_test, BATCH_SIZE)

    # -----------------------------
    # Load Pretrained RBM Model
    # -----------------------------
    rbm = RBM(VISIBLE_UNITS, HIDDEN_UNITS)
    rbm(tf.zeros((1, VISIBLE_UNITS)))  # Force variable build
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "rbm_best.weights.h5")
    if os.path.exists(checkpoint_path):
        rbm.load_weights(checkpoint_path)
        log_message(f"RBM weights loaded from {checkpoint_path}")
    else:
        log_message("No pretrained RBM found. Train the model first.")

    # -----------------------------
    # Generate Synthetic Samples
    # -----------------------------
    synthetic = sample_synthetic(rbm, num_samples=1000, k_steps=K_STEPS, visible_units=VISIBLE_UNITS)
    save_synthetic_samples(synthetic, SYNTHETIC_DIR, prefix="rbm")
    visualize_samples(synthetic, IMG_SHAPE, num_display=16)

    # -----------------------------
    # Evaluate Generated Samples
    # -----------------------------
    eval_results = evaluate_rbm_generation(
        rbm=rbm,
        x_val=x_val,
        img_shape=IMG_SHAPE,
        num_samples=1000,
        k_steps=K_STEPS
    )

    log_message(f"[Evaluation] {eval_results}")

    # -----------------------------
    # Train Classifier on Real-Only
    # -----------------------------
    log_message("Training classifier on real-only data...")
    clf_real = build_classifier(IMG_SHAPE, NUM_CLASSES)
    clf_real = train_classifier(clf_real, train_ds, val_ds, epochs=20, log_dir=os.path.join(LOG_DIR, "clf_real"))
    real_metrics = evaluate_classifier_utility(clf_real, test_ds)
    log_message(f"Classifier Metrics (Real Only): {real_metrics}")

    # -----------------------------
    # Train Classifier on Real + Synthetic
    # -----------------------------
    log_message("Training classifier on real + synthetic data...")

    # Ensure synthetic samples match shape
    if len(synthetic.shape) == 2 and np.prod(IMG_SHAPE) == synthetic.shape[1]:
        x_synth = synthetic.reshape(-1, *IMG_SHAPE)
    else:
        raise ValueError(f"Expected shape ({-1}, {np.prod(IMG_SHAPE)}), but got {synthetic.shape}")

    y_synth = y_train[:len(x_synth)]

    x_train_reshaped = x_train.reshape(-1, *IMG_SHAPE)
    x_combined = np.concatenate([x_train_reshaped, x_synth], axis=0)

    y_combined = np.concatenate([y_train, y_synth], axis=0)

    combined_ds = prepare_tf_dataset(x_combined.reshape(-1, *IMG_SHAPE), y_combined, BATCH_SIZE)

    clf_combined = build_classifier(IMG_SHAPE, NUM_CLASSES)
    clf_combined = train_classifier(clf_combined, combined_ds, val_ds, epochs=20, log_dir=os.path.join(LOG_DIR, "clf_combined"))
    combined_metrics = evaluate_classifier_utility(clf_combined, test_ds)
    log_message(f"Classifier Metrics (Real + Synthetic): {combined_metrics}")

    # -----------------------------
    # Log Final Accuracy Comparison
    # -----------------------------
    final_comparison_path = os.path.join(BASE_DIR, "rbm_classifier_comparison.txt")
    log_comparison_results({
        "real_only": real_metrics["accuracy"],
        "real_plus_synthetic": combined_metrics["accuracy"]
    }, output_path=final_comparison_path)
