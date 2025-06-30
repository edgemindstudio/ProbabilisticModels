# =====================================
# SECTION 1: Imports and Initial Setup
# =====================================

import os
import yaml
import random
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
import tensorflow as tf
from tensorflow.keras import models, layers, losses, optimizers, callbacks


# ==============================
# Set Random Seeds
# ==============================
def set_random_seeds(seed=42):
    """Ensure reproducibility across NumPy and random."""
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
NUM_CLASSES = config["NUM_CLASSES"]
RUN_MODE = config.get("mode", "train")
VERBOSE = config.get("verbose", True)
PATIENCE = config.get("patience", 10)

# ==============================
# Define Output Paths
# ==============================
EXPERIMENT_NAME = "gaussian_mixture_model"
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


def log_comparison_results(results_dict, output_path):
    with open(output_path, "w") as f:
        for key, value in results_dict.items():
            f.write(f"{key}: {value}\n")



# =====================================
# SECTION 2A: Data Loading and Preprocessing
# =====================================

def load_malware_dataset(data_path, img_shape, num_classes, val_fraction=0.5):
    """
    Load and preprocess malware image data for Gaussian Mixture Model.

    Args:
        data_path (str): Path to dataset directory containing .npy files
        img_shape (tuple): Original shape of images (H, W, C)
        num_classes (int): Number of unique labels
        val_fraction (float): Fraction of test data to reserve for validation

    Returns:
        Tuple: (x_train, y_train), (x_val, y_val), (x_test, y_test)
    """
    # Load datasets
    x_train = np.load(os.path.join(data_path, "train_data.npy"))
    y_train = np.load(os.path.join(data_path, "train_labels.npy"))
    x_test = np.load(os.path.join(data_path, "test_data.npy"))
    y_test = np.load(os.path.join(data_path, "test_labels.npy"))

    # Normalize to [0, 1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    # Flatten the images
    x_train = x_train.reshape((-1, np.prod(img_shape)))
    x_test = x_test.reshape((-1, np.prod(img_shape)))

    # One-hot encode the labels (optional for classifier utility)
    y_train = np.eye(num_classes)[y_train]
    y_test = np.eye(num_classes)[y_test]

    # Split test into validation and test
    split_idx = int(len(x_test) * val_fraction)
    x_val, y_val = x_test[:split_idx], y_test[:split_idx]
    x_test, y_test = x_test[split_idx:], y_test[split_idx:]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)



# =====================================
# SECTION 2B: Evaluation Metric Placeholders
# =====================================

# ------------------------------
# Metric 1: Fréchet Inception Distance (FID)
# ------------------------------
def calculate_fid(real_images, generated_images):
    """
    Compute Fréchet Inception Distance (FID) between real and generated samples.

    Args:
        real_images (ndarray): Real images in [0,1], shape [N, H, W, C]
        generated_images (ndarray): Generated images in [0,1], shape [N, H, W, C]

    Returns:
        float: FID score
    """
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


# ------------------------------
# Metric 2: JS & KL Divergence
# ------------------------------
def js_divergence(p, q):
    """Compute Jensen-Shannon divergence between two distributions."""
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def kl_divergence(p, q):
    """Compute Kullback-Leibler divergence between two distributions."""
    return entropy(p, q)


# ------------------------------
# Metric 3: Classifier Evaluation
# ------------------------------
def evaluate_classifier_metrics(y_true, y_pred, average="macro"):
    """
    Compute accuracy, precision, recall, F1, and confusion matrix.

    Args:
        y_true (ndarray): True labels (int)
        y_pred (ndarray): Predicted labels (int)

    Returns:
        dict: Classification metrics
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average),
        "recall": recall_score(y_true, y_pred, average=average),
        "f1": f1_score(y_true, y_pred, average=average),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }


# ------------------------------
# Metric 4: Sample Diversity Score
# ------------------------------
def compute_sample_diversity(samples):
    """
    Compute diversity score using variance across dimensions.

    Args:
        samples (ndarray): Synthetic data, flattened [N, D]

    Returns:
        float: Mean feature-wise variance
    """
    return float(np.mean(np.var(samples, axis=0)))



# =====================================
# SECTION 3: Gaussian Mixture Model (GMM) Definition and Fitting
# =====================================

from sklearn.mixture import GaussianMixture

def train_gmm_model(x_train, num_components=10, random_seed=42):
    """
    Fit a Gaussian Mixture Model (GMM) to the training data.

    Args:
        x_train (ndarray): Flattened training samples [N, D]
        num_components (int): Number of Gaussian components (clusters)
        random_seed (int): Seed for reproducibility

    Returns:
        GaussianMixture: Trained GMM model
    """
    log_message(f"Fitting GMM with {num_components} components on {x_train.shape[0]} samples.")
    gmm = GaussianMixture(
        n_components=num_components,
        covariance_type='full',
        random_state=random_seed,
        max_iter=200,
        verbose=1
    )
    gmm.fit(x_train)
    log_message(f"Finished training GMM. Converged: {gmm.converged_}, Iterations: {gmm.n_iter_}")
    return gmm


def sample_from_gmm(gmm_model, num_samples):
    """
    Sample new data points from a trained GMM.

    Args:
        gmm_model (GaussianMixture): Trained GMM object
        num_samples (int): Number of samples to generate

    Returns:
        ndarray: Synthetic samples [N, D]
    """
    samples, _ = gmm_model.sample(num_samples)
    samples = np.clip(samples, 0.0, 1.0)
    return samples



# =====================================
# SECTION 4: GMM Synthetic Sample Saving and Visualization
# =====================================

def reshape_gmm_samples(samples, img_shape):
    """
    Reshape flattened GMM samples back to image format.

    Args:
        samples (ndarray): Flattened synthetic samples [N, D]
        img_shape (tuple): Desired image shape (H, W, C)

    Returns:
        ndarray: Reshaped image samples [N, H, W, C]
    """
    return samples.reshape((-1, *img_shape))


def save_synthetic_samples(samples, output_dir, prefix="gmm"):
    """
    Save synthetic GMM samples to .npy files.

    Args:
        samples (ndarray): Reshaped samples [N, H, W, C]
        output_dir (str): Directory to save files
        prefix (str): File name prefix
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, sample in enumerate(samples):
        np.save(os.path.join(output_dir, f"{prefix}_{i}.npy"), sample)


def visualize_gmm_samples(samples, num_display=9, save_path=None):
    """
    Visualize a grid of generated GMM samples.

    Args:
        samples (ndarray): Image samples [N, H, W, C]
        num_display (int): Number of images to display
    """
    grid_size = int(np.sqrt(num_display))
    plt.figure(figsize=(grid_size * 2, grid_size * 2))

    for i in range(num_display):
        plt.subplot(grid_size, grid_size, i + 1)
        img = samples[i]
        if img.shape[-1] == 1:
            plt.imshow(img.squeeze(), cmap='gray')
        else:
            plt.imshow(img)
        plt.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()



# =====================================
# SECTION 5: GMM Evaluation of Generated Samples
# =====================================

def evaluate_generated_samples(real_images, generated_images, alpha=1e-6):
    """
    Evaluate GMM-generated samples using FID, JS, KL, and diversity metrics.

    Args:
        real_images (ndarray): Real image samples [N, H, W, C]
        generated_images (ndarray): Generated image samples [N, H, W, C]
        alpha (float): Smoothing for histogram calculations

    Returns:
        dict: Dictionary of evaluation metric results
    """
    # Resize to (299, 299, 3) for InceptionV3
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

    # FID
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean))

    # JS & KL divergence (pixel histogram)
    def get_histogram(imgs):
        return np.histogram(imgs.flatten(), bins=256, range=(0, 1), density=True)[0] + alpha

    p_real = get_histogram(real_images)
    p_fake = get_histogram(generated_images)
    js = js_divergence(p_real, p_fake)
    kl = kl_divergence(p_real, p_fake)

    # Diversity score
    diversity = compute_sample_diversity(generated_images)

    return {
        "FID": fid,
        "JS_Divergence": js,
        "KL_Divergence": kl,
        "Diversity_Score": diversity
    }



# =====================================
# SECTION 6: Classifier Utility Evaluation for GMM
# =====================================

def build_classifier(input_shape, num_classes):
    """
    Build a CNN classifier model.

    Args:
        input_shape (tuple): Shape of input images (H, W, C)
        num_classes (int): Number of output classes

    Returns:
        tf.keras.Model: Compiled classifier model
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
    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.CategoricalCrossentropy(),
        metrics=['accuracy']
    )
    return model


def train_classifier(model, x_train, y_train, x_val, y_val, log_dir, epochs=20):
    """
    Train a CNN classifier with early stopping and TensorBoard support.

    Args:
        model (tf.keras.Model): Compiled classifier model
        x_train, y_train: Training set (one-hot)
        x_val, y_val: Validation set
        log_dir (str): Path to TensorBoard log directory
        epochs (int): Max training epochs

    Returns:
        tf.keras.Model: Trained classifier
    """
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
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
    Evaluate trained classifier on test set.

    Args:
        model: Trained Keras model
        x_test, y_test: Test set (one-hot)

    Returns:
        dict: Classifier evaluation metrics
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
# SECTION 7: Main Runner Script
# =====================================

if __name__ == "__main__":
    # -----------------------------
    # Device Configuration
    # -----------------------------
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        device = "GPU"
    else:
        device = "CPU"
    log_message(f"Using device: {device}")

    # -----------------------------
    # Load Dataset
    # -----------------------------
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_malware_dataset(
        data_path=DATA_PATH,
        img_shape=IMG_SHAPE,
        num_classes=NUM_CLASSES,
        val_fraction=0.5
    )

    x_train_flat = x_train.reshape((-1, np.prod(IMG_SHAPE)))
    x_val_flat = x_val.reshape((-1, np.prod(IMG_SHAPE)))
    x_test_flat = x_test.reshape((-1, np.prod(IMG_SHAPE)))

    # Reshape back to image shape for CNN
    x_train_cnn = x_train.reshape((-1, *IMG_SHAPE))
    x_val_cnn = x_val.reshape((-1, *IMG_SHAPE))
    x_test_cnn = x_test.reshape((-1, *IMG_SHAPE))

    # -----------------------------
    # Train Gaussian Mixture Model
    # -----------------------------
    gmm_model = train_gmm_model(
        x_train_flat,
        num_components=config["GMM_COMPONENTS"]
    )

    # -----------------------------
    # Generate and Save Synthetic Samples
    # -----------------------------
    synthetic_flat = sample_from_gmm(gmm_model, num_samples=1000)
    synthetic_images = synthetic_flat.reshape((-1, *IMG_SHAPE))
    save_synthetic_samples(synthetic_images, output_dir=SYNTHETIC_DIR, prefix="gmm")

    visualize_gmm_samples(synthetic_images, num_display=16, save_path=os.path.join(BASE_DIR, "gmm_generated_samples.png"))

    # -----------------------------
    # Evaluate Generation Quality
    # -----------------------------
    eval_results = evaluate_generated_samples(
        real_images=x_val[:1000].reshape((-1, *IMG_SHAPE)),
        generated_images=synthetic_images[:1000]
    )

    log_message(f"GMM Evaluation Metrics: {eval_results}")

    # -----------------------------
    # Classifier Utility Evaluation
    # -----------------------------
    # (1) Real-only classifier
    clf_real = build_classifier(input_shape=IMG_SHAPE, num_classes=NUM_CLASSES)
    clf_real = train_classifier(clf_real, x_train_cnn, y_train, x_val_cnn, y_val, LOG_DIR, epochs=20)

    real_metrics = evaluate_classifier(clf_real, x_test_cnn, y_test)

    log_message(f"[Classifier Real Only] {real_metrics}")

    # (2) Real + Synthetic classifier
    y_synth = y_train[:len(synthetic_images)]
    x_combined = np.concatenate([x_train_cnn, synthetic_images], axis=0)

    y_combined = np.concatenate([y_train, y_synth], axis=0)

    clf_combined = build_classifier(input_shape=IMG_SHAPE, num_classes=NUM_CLASSES)
    x_combined_cnn = x_combined.reshape((-1, *IMG_SHAPE))
    clf_combined = train_classifier(clf_combined, x_combined_cnn, y_combined, x_val_cnn, y_val, LOG_DIR, epochs=20)

    combined_metrics = evaluate_classifier(clf_combined, x_test_cnn, y_test)

    log_message(f"[Classifier Real + Synthetic] {combined_metrics}")

    # -----------------------------
    # Save Accuracy Comparison
    # -----------------------------
    log_comparison_results({
        "real_only": real_metrics["accuracy"],
        "real_plus_synthetic": combined_metrics["accuracy"]
    }, output_path=os.path.join(BASE_DIR, "gmm_classifier_comparison.txt"))
