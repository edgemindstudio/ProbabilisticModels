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


# =====================================
# SECTION 3: Define Masked Convolution Layer
# =====================================

class MaskedConv2D(layers.Layer):
    """
    Custom Masked Convolution Layer for PixelCNN.
    Supports mask type 'A' for the first layer and 'B' for subsequent layers.
    """
    def __init__(self, filters, kernel_size, mask_type, **kwargs):
        super(MaskedConv2D, self).__init__(**kwargs)
        assert mask_type in ['A', 'B'], "mask_type must be 'A' or 'B'"
        self.filters = filters
        self.kernel_size = kernel_size
        self.mask_type = mask_type
        self.conv = layers.Conv2D(filters, kernel_size, padding='same', use_bias=False)

    def build(self, input_shape):
        self.conv.build(input_shape)
        kh, kw = self.kernel_size
        in_channels = input_shape[-1]
        out_channels = self.filters

        # Initialize mask with ones
        mask = np.ones((kh, kw, in_channels, out_channels), dtype=np.float32)

        center_h = kh // 2
        center_w = kw // 2

        # Apply mask rules
        mask[center_h, center_w + (self.mask_type == 'B'):, :, :] = 0
        mask[center_h + 1:, :, :, :] = 0

        self.mask = tf.constant(mask, dtype=tf.float32)

    def call(self, inputs):
        masked_kernel = self.conv.kernel * self.mask
        outputs = tf.nn.conv2d(inputs, masked_kernel, strides=1, padding="SAME")
        return outputs


# =====================================
# SECTION 4A: Gated Activation Units (GAUs)
# =====================================
def gated_activation_unit(x):
    """
    Implements the Gated Activation Unit from PixelCNN++:
    splits input into two parts and applies tanh/sigmoid gating.
    """
    x_tanh, x_sigmoid = tf.split(x, num_or_size_splits=2, axis=-1)
    return tf.math.tanh(x_tanh) * tf.math.sigmoid(x_sigmoid)


# =====================================
# SECTION 4B: Build the Conditional PixelCNN Model
# =====================================
def build_conditional_pixelcnn(img_shape, num_classes, num_filters=64, num_layers=7, dropout_rate=0.1):
    """
    Build an enhanced Conditional PixelCNN model with residual connections,
    normalization, optional dropout, and Gated Activation Units.
    """
    input_image = layers.Input(shape=img_shape, name="input_image")
    label_input = layers.Input(shape=(num_classes,), name="label_input")

    # Condition label to match image shape
    label_map = layers.Dense(np.prod(img_shape), activation="relu")(label_input)
    label_map = layers.Reshape(img_shape)(label_map)

    # Concatenate label map with input
    x = layers.Concatenate(axis=-1)([input_image, label_map])

    # First Masked Conv2D (Type A) - doubles filters for gated unit
    x = MaskedConv2D(filters=2 * num_filters, kernel_size=(7, 7), mask_type='A', name="masked_conv_A")(x)
    x = gated_activation_unit(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Stack of Masked Conv2D (Type B) with residual connections
    for i in range(num_layers):
        residual = x
        x = MaskedConv2D(filters=2 * num_filters, kernel_size=(3, 3), mask_type='B', name=f"masked_conv_B_{i}")(x)
        x = gated_activation_unit(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Add()([x, residual])  # Residual connection

    # Final layer reduces back to 1 output channel
    output = layers.Conv2D(1, kernel_size=1, activation="sigmoid", name="output")(x)

    return models.Model(inputs=[input_image, label_input], outputs=output, name="Conditional_PixelCNN")


# =====================================
# Define Loss Function and Optimizer
# =====================================

# Loss function for pixel-wise prediction (binary output in [0, 1])
bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# Optimizer (use legacy for better performance on M1/M2 Macs)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=LR, beta_1=BETA_1)


# =====================================
# SECTION 5A: Training Step and Sampling Utilities
# =====================================

@tf.function
def train_step(model, x_batch, y_batch):
    """
    Performs one training step for the PixelCNN model.

    Args:
        model: The conditional PixelCNN model.
        x_batch: Input images (B, H, W, C), values in [0, 1].
        y_batch: One-hot encoded labels (B, num_classes).

    Returns:
        Scalar loss value for the batch.
    """
    with tf.GradientTape() as tape:
        predictions = model([x_batch, y_batch], training=True)
        loss = bce_loss_fn(x_batch, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# =====================================
# SECTION 5B: Full Training Loop (with Logging and Early Stopping)
# =====================================

def train_pixelcnn(model, train_dataset, val_dataset, epochs, summary_writer, checkpoint_dir, patience=10):
    """
    Train the conditional PixelCNN model with early stopping and logging.

    Args:
        model: The PixelCNN model.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        epochs: Total number of training epochs.
        summary_writer: TensorBoard writer.
        checkpoint_dir: Directory to save best model weights.
        patience: Number of epochs to wait for improvement before stopping.

    Returns:
        None
    """
    log_message("Starting PixelCNN training...")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in trange(epochs, desc="Epochs"):
        train_losses = []

        for x_batch, y_batch in train_dataset:
            loss = train_step(model, x_batch, y_batch)
            train_losses.append(loss.numpy())

        avg_train_loss = np.mean(train_losses)

        # Validation phase
        val_losses = []
        for val_x, val_y in val_dataset:
            val_preds = model([val_x, val_y], training=False)
            val_loss = bce_loss_fn(val_x, val_preds)
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
            model.save_weights(os.path.join(checkpoint_dir, "pixelcnn_best.h5"))
        else:
            patience_counter += 1
            log_message(f"Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                log_message("Early stopping triggered.")
                break

        # Optional: periodic checkpoint saving
        if (epoch + 1) % 50 == 0:
            model.save_weights(os.path.join(checkpoint_dir, f"pixelcnn_epoch_{epoch+1}.h5"))


# =====================================
# SECTION 6: Autoregressive Sampling Function
# =====================================

def sample_from_pixelcnn(model, num_samples, img_shape, num_classes):
    """
    Generate synthetic images using autoregressive sampling.

    Args:
        model: Trained PixelCNN model.
        num_samples: Number of samples to generate.
        img_shape: Tuple of image dimensions (H, W, C).
        num_classes: Total number of class labels.

    Returns:
        Tuple: (generated_images, labels)
    """
    height, width, channels = img_shape
    generated_images = np.zeros((num_samples, height, width, channels), dtype=np.float32)

    # Generate random class labels and convert to one-hot
    labels = np.random.randint(0, num_classes, size=num_samples)
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes)

    # Generate image pixel-by-pixel
    for i in range(height):
        for j in range(width):
            for c in range(channels):
                logits = model.predict([generated_images, one_hot_labels], verbose=0)
                probs = logits[:, i, j, c]
                sampled = tf.cast(tf.random.uniform(shape=probs.shape) < probs, tf.float32)
                generated_images[:, i, j, c] = sampled.numpy()

    return generated_images, labels


# =====================================
# SECTION 7: Generate and Visualize Conditional Samples
# =====================================

def generate_and_plot_pixelcnn_samples(model, img_shape, num_classes, samples_per_class=1, save_path=None):
    """
    Generate and visualize PixelCNN samples conditioned on each class label.

    Args:
        model: Trained PixelCNN model.
        img_shape: Shape of generated images.
        num_classes: Number of class labels.
        samples_per_class: Number of samples to generate per class.
        save_path: Optional path to save the plot.

    Returns:
        None
    """
    total_samples = num_classes * samples_per_class
    generated_images, labels = sample_from_pixelcnn(model, total_samples, img_shape, num_classes)

    # Clip to ensure image values are in [0, 1]
    generated_images = np.clip(generated_images, 0.0, 1.0)

    # Plot the generated samples
    plt.figure(figsize=(samples_per_class * 2, num_classes * 2))
    for i in range(total_samples):
        plt.subplot(num_classes, samples_per_class, i + 1)
        plt.imshow(generated_images[i].squeeze(), cmap='gray')
        plt.title(f"Class {labels[i]}")
        plt.axis("off")

    if save_path:
        plt.savefig(save_path)
    plt.close()


# =====================================
# SECTION 8: Compute FID Score
# =====================================

def calculate_fid(real_images, generated_images):
    """
    Compute the Fréchet Inception Distance (FID) between real and generated images.

    Args:
        real_images: Array of real images in [0, 1], shape (N, H, W, 1).
        generated_images: Array of generated images in [0, 1], shape (N, H, W, 1).

    Returns:
        FID score (float).
    """
    # Resize to 299x299 and convert grayscale to RGB
    real_resized = tf.image.resize(real_images, (299, 299))
    fake_resized = tf.image.resize(generated_images, (299, 299))

    real_rgb = tf.image.grayscale_to_rgb(real_resized)
    fake_rgb = tf.image.grayscale_to_rgb(fake_resized)

    # Preprocess for InceptionV3
    real_rgb = preprocess_input(real_rgb)
    fake_rgb = preprocess_input(fake_rgb)

    # Load pre-trained InceptionV3 (without classification head)
    inception = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

    # Extract features
    act1 = inception.predict(real_rgb, verbose=0)
    act2 = inception.predict(fake_rgb, verbose=0)

    # Compute statistics
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)

    # Compute FID
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1 @ sigma2)

    # Handle numerical instability
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)

    return fid


# =====================================
# SECTION 9: Build Classifier for Evaluation
# =====================================

def build_classifier(input_shape=(40, 40, 1), num_classes=9):
    """
    Build a simple CNN classifier for evaluating real vs. synthetic data.

    Args:
        input_shape: Shape of the input images.
        num_classes: Number of output classes.

    Returns:
        A compiled tf.keras.Sequential classifier model.
    """
    model = models.Sequential(name="Autoregressive_Eval_Classifier")
    model.add(layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# =====================================
# SECTION 10: Train & Evaluate Real-Only Baseline Classifier
# =====================================

def train_real_classifier(train_data, val_data, input_shape, num_classes, checkpoint_path, epochs=30):
    """
    Train a CNN classifier using only real data for evaluation benchmarking.

    Args:
        train_data: TensorFlow dataset with real training data.
        val_data: TensorFlow dataset with real validation data.
        input_shape: Input image shape.
        num_classes: Number of output classes.
        checkpoint_path: Path to save best weights.
        epochs: Number of training epochs.

    Returns:
        Trained classifier model.
    """
    log_message("Training real-only classifier for evaluation...")

    classifier = build_classifier(input_shape=input_shape, num_classes=num_classes)

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


# =====================================
# SECTION 11: Generate Synthetic Samples per Class
# =====================================

def generate_synthetic_per_class_ar(model, num_classes, samples_per_class, img_shape, save_dir=SYNTHETIC_DIR):
    """
    Generate and save synthetic samples for each class using the autoregressive model.

    Args:
        model: Trained PixelCNN model.
        num_classes: Total number of distinct classes.
        samples_per_class: Number of samples to generate per class.
        img_shape: Shape of output images (H, W, C).
        save_dir: Root directory to store generated samples.

    Returns:
        None
    """
    log_message(f"Generating {samples_per_class} autoregressive samples for each of {num_classes} classes...")

    os.makedirs(save_dir, exist_ok=True)

    for class_id in range(num_classes):
        class_dir = os.path.join(save_dir, f"class_{class_id}")
        os.makedirs(class_dir, exist_ok=True)

        for i in range(samples_per_class):
            print(f"[INFO] Generating sample {i + 1}/{samples_per_class} for class {class_id}")
            label_onehot = tf.keras.utils.to_categorical([class_id], num_classes=num_classes)
            sample, _ = sample_from_pixelcnn(model, 1, img_shape, num_classes)

            # Rescale to [0, 1] if needed
            clipped = tf.clip_by_value(sample, 0.0, 1.0).numpy().squeeze()
            save_path = os.path.join(class_dir, f"sample_{i}.npy")
            np.save(save_path, clipped)

    log_message("Completed generation of all synthetic class samples.")


# =====================================
# SECTION 12: Load Synthetic Samples from Disk
# =====================================

def load_synthetic_samples_ar(root_dir, num_classes, samples_per_class=100, img_shape=(40, 40, 1)):
    """
    Load saved synthetic samples (.npy format) from disk.

    Args:
        root_dir: Path to the directory containing per-class sample folders.
        num_classes: Total number of classes.
        samples_per_class: Number of samples to load per class.
        img_shape: Expected shape of each sample.

    Returns:
        Tuple of (x_synthetic, y_synthetic_onehot)
    """
    log_message(f"Loading autoregressive synthetic samples from '{root_dir}'...")

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


# =====================================
# SECTION 13A: Train Classifier on Real Data Only
# =====================================

def train_real_only_classifier_ar(train_data, val_data, input_shape, num_classes, epochs=20, log_path="ar_classifier_real_only.h5"):
    """
    Train a CNN classifier using only real training data (baseline benchmark).

    Args:
        train_data: TensorFlow dataset containing real images and labels.
        val_data: TensorFlow dataset for validation.
        input_shape: Input shape of images.
        num_classes: Number of classification labels.
        epochs: Total training epochs.
        log_path: Path to save the trained model.

    Returns:
        Trained classifier model.
    """
    log_message("Training classifier on real data only...")

    model = build_classifier(input_shape=input_shape, num_classes=num_classes)

    # Define EarlyStopping
    early_stopping_cb = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        verbose=2,
        callbacks=[early_stopping_cb]  # Add callback here
    )

    model.save(log_path)

    log_message(f"Saved real-only classifier to {log_path}")

    return model, history


# =====================================
# SECTION 13B: Train Classifier on Real + Synthetic Data
# =====================================

def train_classifier_on_real_plus_synthetic_ar(
    real_train_dataset,
    synthetic_data,
    val_data,
    input_shape,
    num_classes,
    batch_size,
    epochs=20,
    save_path="classifier_combined_ar.h5"
):
    """
    Train a CNN classifier using both real and synthetic data.

    Args:
        real_train_dataset: tf.data.Dataset with real training data.
        synthetic_data: Tuple (x_synth, y_synth) from autoregressive generator.
        val_data: Validation dataset.
        input_shape: Input image shape.
        num_classes: Number of output classes.
        batch_size: Batch size for training.
        epochs: Total number of training epochs.
        save_path: File path to save trained model.

    Returns:
        Trained classifier model.
    """
    log_message("Training classifier on real + synthetic data...")

    # Extract real data into numpy arrays
    x_real, y_real = [], []
    for batch in real_train_dataset:
        x_real.append(batch[0].numpy())
        y_real.append(batch[1].numpy())
    x_real = np.concatenate(x_real, axis=0)
    y_real = np.concatenate(y_real, axis=0)

    # Combine real and synthetic data
    x_synth, y_synth = synthetic_data
    x_combined = np.concatenate([x_real, x_synth], axis=0)
    y_combined = np.concatenate([y_real, y_synth], axis=0)

    # Create tf.data.Dataset from combined data
    combined_dataset = tf.data.Dataset.from_tensor_slices((x_combined, y_combined))
    combined_dataset = combined_dataset.shuffle(2048).batch(batch_size).prefetch(tf.data.AUTOTUNE)


    # Build and train classifier
    model = build_classifier(input_shape=input_shape, num_classes=num_classes)

    # Define EarlyStopping
    early_stopping_cb = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        combined_dataset,
        validation_data=val_data,
        epochs=epochs,
        verbose=2,
        callbacks=[early_stopping_cb]  # Add callback here
    )

    model.save(save_path)

    log_message(f"Saved real + synthetic classifier to {save_path}")

    return model, history



# =====================================
# SECTION 14: Evaluate Classifier on Real vs. Real+Synthetic
# =====================================

def evaluate_classifier_on_real_vs_combined_ar(classifier_real, classifier_combined, x_test, y_test):
    """
    Evaluate two classifiers:
    - One trained on real data only.
    - One trained on real + synthetic data.

    Args:
        classifier_real: Trained model using only real data.
        classifier_combined: Trained model using real + synthetic data.
        x_test: Test images.
        y_test: Test labels (one-hot encoded).

    Returns:
        Dictionary with accuracy comparison results.
    """
    log_message("Evaluating classifier trained on real data only...")
    real_eval = classifier_real.evaluate(x_test, y_test, verbose=0)
    real_accuracy = real_eval[1]

    log_message("Evaluating classifier trained on real + synthetic data...")
    combined_eval = classifier_combined.evaluate(x_test, y_test, verbose=0)
    combined_accuracy = combined_eval[1]

    log_message(f"Real-Only Accuracy:        {real_accuracy:.4f}")
    log_message(f"Real+Synthetic Accuracy:   {combined_accuracy:.4f}")

    return {
        "real_only": real_accuracy,
        "real_plus_synthetic": combined_accuracy
    }


# =====================================
# SECTION 15A: Log Comparison Results
# =====================================

def log_comparison_results_ar(results_dict, output_path="ar_comparison_results.txt"):
    """
    Log the accuracy comparison between real-only and real+synthetic classifiers.

    Args:
        results_dict: Dictionary containing accuracy values.
        output_path: File path to save the log.

    Returns:
        None
    """
    log_message("Logging classifier comparison results for Autoregressive Model...")

    with open(output_path, "w") as f:
        f.write("=== Autoregressive Model Classifier Evaluation ===\n")
        f.write(f"Real-Only Accuracy:        {results_dict['real_only']:.4f}\n")
        f.write(f"Real+Synthetic Accuracy:   {results_dict['real_plus_synthetic']:.4f}\n")

    log_message(f"Comparison results saved to: {output_path}")


# =====================================
# SECTION 15B: Save Training & Generation Plots
# =====================================

def save_metric_plot(history, metrics, title, ylabel, filename):
    """
    Saves a training plot (e.g., loss, accuracy) for specified metrics.

    Args:
        history: Keras History object from model.fit().
        metrics: List of metric names to plot (e.g., ["loss", "val_loss"]).
        title: Title of the plot.
        ylabel: Label for the y-axis.
        filename: Output file name to save the plot.
    """
    plt.figure(figsize=(8, 5))
    for metric in metrics:
        plt.plot(history.history[metric], label=metric.replace("_", " ").title())

    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# =====================================
# SECTION 16: Main Runner Script – Autoregressive Model
# =====================================

if __name__ == "__main__":
    log_message("Initializing Autoregressive Model...")

    # -----------------------------------------------------
    # Load training, validation, and test datasets
    # -----------------------------------------------------
    train_dataset, val_dataset, test_dataset = create_datasets(BATCH_SIZE)

    # -----------------------------------------------------
    # Build the Conditional PixelCNN model
    # -----------------------------------------------------
    ar_model = build_conditional_pixelcnn(IMG_SHAPE, NUM_CLASSES)

    # -----------------------------------------------------
    # Compile the model with optimizer and binary cross-entropy loss
    # -----------------------------------------------------
    ar_model.compile(optimizer=optimizer, loss=bce_loss_fn)

    # -----------------------------------------------------
    # Train or evaluate model based on the RUN_MODE flag
    # -----------------------------------------------------
    if RUN_MODE == "train":
        log_message("Starting PixelCNN training...")
        train_pixelcnn(
            model=ar_model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=EPOCHS,
            summary_writer=summary_writer,
            checkpoint_dir=CHECKPOINT_DIR,
            patience=10
        )
        ar_model.save_weights(os.path.join(CHECKPOINT_DIR, "ar_final.h5"))

    elif RUN_MODE == "eval_only":
        log_message("Loading best saved model for evaluation...")
        ar_model.load_weights(os.path.join(CHECKPOINT_DIR, "pixelcnn_best.h5"))

    # -----------------------------------------------------
    # Generate synthetic samples per class using the trained model
    # -----------------------------------------------------
    generate_synthetic_per_class_ar(
        model=ar_model,
        img_shape=IMG_SHAPE,
        num_classes=NUM_CLASSES,
        samples_per_class=5
    )

    # -----------------------------------------------------
    # Load synthetic samples from disk
    # -----------------------------------------------------
    x_synth, y_synth = load_synthetic_samples_ar(
        root_dir=SYNTHETIC_DIR,
        num_classes=NUM_CLASSES,
        samples_per_class=100,
        img_shape=IMG_SHAPE
    )

    # =====================================
    # SECTION 15B: FID Evaluation (Autoregressive Model)
    # =====================================
    log_message("Evaluating FID score for autoregressive synthetic samples...")

    # Load real validation images (truncate to match synthetic sample size)
    x_real_fid = []
    for batch in val_dataset:
        x_real_fid.append(batch[0])
    x_real_fid = tf.concat(x_real_fid, axis=0)
    x_real_fid = tf.convert_to_tensor(x_real_fid[:x_synth.shape[0]])

    # Convert synthetic images to tensor
    x_fake_fid = tf.convert_to_tensor(x_synth)

    # Compute Fréchet Inception Distance (FID)
    fid_score_ar = calculate_fid(x_real_fid, x_fake_fid)
    log_message(f"FID Score (Conditional AR Model): {fid_score_ar:.4f}")

    # Append FID result to comparison log file
    with open("ar_comparison_results.txt", "a") as f:
        f.write(f"FID Score (Autoregressive): {fid_score_ar:.4f}\n")

    # Log FID score to TensorBoard
    with summary_writer.as_default():
        tf.summary.scalar("FID_Score_Autoregressive", fid_score_ar, step=0)
        summary_writer.flush()

    # =====================================
    # SECTION 17: Classifier Training and Evaluation
    # =====================================

    # Train classifier on real data only
    classifier_real, history_real = train_real_only_classifier_ar(
        train_data=train_dataset,
        val_data=val_dataset,
        input_shape=IMG_SHAPE,
        num_classes=NUM_CLASSES,
        epochs=20,
        log_path="classifier_real_ar.h5"
    )

    # Train classifier on combined real + synthetic data
    classifier_combined, history_combined = train_classifier_on_real_plus_synthetic_ar(
        real_train_dataset=train_dataset,
        synthetic_data=(x_synth, y_synth),
        val_data=val_dataset,
        input_shape=IMG_SHAPE,
        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        epochs=20,
        save_path="classifier_combined_ar.h5"
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
    results = evaluate_classifier_on_real_vs_combined_ar(
        classifier_real, classifier_combined, x_test, y_test
    )

    # =====================================
    # SECTION 17: Plot Graphs
    # =====================================

    # For Loss
    save_metric_plot(
        history=history_real,
        metrics=["loss", "val_loss"],
        title="Real-Only Classifier Loss",
        ylabel="Loss",
        filename="real_only_loss.png"
    )

    save_metric_plot(
        history=history_combined,
        metrics=["loss", "val_loss"],
        title="Real+Synthetic Classifier Loss",
        ylabel="Loss",
        filename="combined_loss.png"
    )

    # For Accuracy
    save_metric_plot(
        history=history_real,
        metrics=["accuracy", "val_accuracy"],
        title="Real-Only Classifier Accuracy",
        ylabel="Accuracy",
        filename="real_only_accuracy.png"
    )

    save_metric_plot(
        history=history_combined,
        metrics=["accuracy", "val_accuracy"],
        title="Real+Synthetic Classifier Accuracy",
        ylabel="Accuracy",
        filename="combined_accuracy.png"
    )

    # Log evaluation comparison results
    log_comparison_results_ar(results)
