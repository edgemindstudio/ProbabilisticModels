# -------------------------------------
# SECTION 1: Imports and Initial Setup (Conditional DCGAN)
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

# === Conditional DCGAN CONFIG ===
EXPERIMENT_NAME = "conditional_dcgan"
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

with open(os.path.join(LOG_DIR, "hyperparameters.txt"), "w") as f:
    f.write(f"IMG_SHAPE={IMG_SHAPE}, LATENT_DIM={LATENT_DIM}, NUM_CLASSES={NUM_CLASSES}, "
            f"BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, LR={LR}, BETA_1={BETA_1}\n")

# Dataset location
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "USTC-TFC2016_malware")

# Fixed noise and labels for consistent evaluation
FIXED_NOISE = np.random.normal(0, 1, (100, LATENT_DIM))
FIXED_LABELS = np.tile(np.arange(NUM_CLASSES), 100 // NUM_CLASSES).reshape(-1, 1)
FIXED_LABELS_ONEHOT = tf.keras.utils.to_categorical(FIXED_LABELS, NUM_CLASSES)

# Add mode selector for training/evaluation only
RUN_MODE = config.get("mode", "train")  # 'train' or 'eval_only'


# -------------------------------------
# SECTION 2: Load & Normalize Data
# -------------------------------------

# Load training data
train_data = np.load(os.path.join(DATA_PATH, "train_data.npy"), allow_pickle=True)
train_labels = np.load(os.path.join(DATA_PATH, "train_labels.npy"), allow_pickle=True)

# Load test data
test_data = np.load(os.path.join(DATA_PATH, "test_data.npy"), allow_pickle=True)
test_labels = np.load(os.path.join(DATA_PATH, "test_labels.npy"), allow_pickle=True)

# Reshape and normalize to [-1, 1]
train_data = train_data.reshape(-1, *IMG_SHAPE)
test_data = test_data.reshape(-1, *IMG_SHAPE)
train_data = (train_data - 0.5) * 2.0
test_data = (test_data - 0.5) * 2.0

# One-hot encode labels
train_labels = tf.keras.utils.to_categorical(train_labels, NUM_CLASSES)
test_labels = tf.keras.utils.to_categorical(test_labels, NUM_CLASSES)

# Log shape info
log_message(f"Train data shape: {train_data.shape}")
log_message(f"Train labels shape: {train_labels.shape}")
log_message(f"Test data shape: {test_data.shape}")
log_message(f"Test labels shape: {test_labels.shape}")


# -------------------------------------
# SECTION 3: FID Calculation
# -------------------------------------
# InceptionV3 is used to extract features from real and generated images
inception_model = tf.keras.applications.InceptionV3(
    include_top=False, pooling='avg', input_shape=(75, 75, 3)
)

def calculate_fid(real_images, generated_images):
    """Calculate Fréchet Inception Distance (FID)"""
    # Convert from [-1, 1] to [0, 255]
    real_images = (real_images + 1.0) * 127.5
    generated_images = (generated_images + 1.0) * 127.5

    # Resize to match InceptionV3 input shape
    real_images_resized = tf.image.resize(real_images, (75, 75))
    fake_images_resized = tf.image.resize(generated_images, (75, 75))

    # Convert grayscale to RGB by repeating channels
    real_images_resized = tf.repeat(real_images_resized, 3, axis=-1)
    fake_images_resized = tf.repeat(fake_images_resized, 3, axis=-1)

    # Extract features using InceptionV3
    real_activations = inception_model(real_images_resized)
    fake_activations = inception_model(fake_images_resized)

    # Compute mean and covariance
    mu_real = tf.reduce_mean(real_activations, axis=0)
    mu_fake = tf.reduce_mean(fake_activations, axis=0)
    sigma_real = tfp.stats.covariance(real_activations)
    sigma_fake = tfp.stats.covariance(fake_activations)

    # Compute FID score
    ssdiff = tf.reduce_sum(tf.square(mu_real - mu_fake))
    covmean = sqrtm(sigma_real @ sigma_fake)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    if np.isnan(covmean).any():
        return 1e6  # Penalize invalid FID

    fid = ssdiff + tf.linalg.trace(sigma_real + sigma_fake - 2.0 * covmean)
    return fid.numpy()


# -------------------------------------
# SECTION 4: Generator Model (Conditional DCGAN)
# -------------------------------------
def build_generator():
    noise_input = layers.Input(shape=(LATENT_DIM,))
    label_input = layers.Input(shape=(NUM_CLASSES,))

    # Concatenate noise and label
    merged_input = layers.Concatenate()([noise_input, label_input])

    x = layers.Dense(256)(merged_input)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(1024)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(np.prod(IMG_SHAPE), activation='tanh')(x)
    output = layers.Reshape(IMG_SHAPE)(x)

    return models.Model([noise_input, label_input], output, name="Conditional_DCGAN_Generator")


# -------------------------------------
# SECTION 5: Discriminator Model (Conditional DCGAN)
# -------------------------------------
def build_discriminator():
    img_input = layers.Input(shape=IMG_SHAPE)
    label_input = layers.Input(shape=(NUM_CLASSES,))

    # Flatten image and label for concatenation
    img_flat = layers.Flatten()(img_input)
    merged_input = layers.Concatenate()([img_flat, label_input])

    x = layers.Dense(1024)(merged_input)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(512)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(256)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    output = layers.Dense(1, activation='sigmoid')(x)

    return models.Model([img_input, label_input], output, name="Conditional_DCGAN_Discriminator")


# -------------------------------------
# SECTION 6: Loss Functions and Optimizers Setup
# -------------------------------------

# Binary cross-entropy loss for both Generator and Discriminator
bce = tf.keras.losses.BinaryCrossentropy()

def discriminator_loss(real_output, fake_output):
    real_loss = bce(tf.ones_like(real_output), real_output)
    fake_loss = bce(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return bce(tf.ones_like(fake_output), fake_output)

# Optimizers from config
generator_optimizer = Adam(learning_rate=LR, beta_1=BETA_1)
discriminator_optimizer = Adam(learning_rate=LR, beta_1=BETA_1)


# -------------------------------------
# SECTION 7: Training Loop (Conditional DCGAN)
# -------------------------------------
def train_conditional_dcgan():
    log_message("Initializing Conditional Generator...")
    generator = build_generator()

    log_message("Initializing Conditional Discriminator...")
    discriminator = build_discriminator()

    best_fid = float("inf")
    wait = 0

    for epoch in trange(EPOCHS, desc="Training Conditional DCGAN"):
        log_message(f"Epoch {epoch + 1}/{EPOCHS} starting...")
        start_time = time.time()

        # Sample real images
        idx = np.random.randint(0, train_data.shape[0], BATCH_SIZE)
        real_imgs = train_data[idx]
        real_lbls = train_labels[idx]

        # Generate fake images
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        fake_labels = np.random.randint(0, NUM_CLASSES, BATCH_SIZE).reshape(-1, 1)
        fake_labels_onehot = tf.keras.utils.to_categorical(fake_labels, NUM_CLASSES)

        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            fake_imgs = generator([noise, fake_labels_onehot], training=True)

            real_output = discriminator([real_imgs, real_lbls], training=True)
            fake_output = discriminator([fake_imgs, fake_labels_onehot], training=True)

            d_loss = discriminator_loss(real_output, fake_output)
            g_loss = generator_loss(fake_output)

        # Apply gradients
        d_grads = disc_tape.gradient(d_loss, discriminator.trainable_variables)
        g_grads = gen_tape.gradient(g_loss, generator.trainable_variables)

        discriminator_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
        generator_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

        fid = None

        # Every 100 epochs
        if (epoch + 1) % 100 == 0:
            fid = calculate_fid(real_imgs[:100], fake_imgs[:100])
            log_message(f"Epoch {epoch + 1} — FID: {fid:.4f} | G Loss: {g_loss:.4f} | D Loss: {d_loss:.4f}")

            improved = False
            if fid < best_fid:
                improved = True
                best_fid = fid
                wait = 0
                generator.save_weights(os.path.join(CHECKPOINT_DIR, "best_generator.h5"))
                discriminator.save_weights(os.path.join(CHECKPOINT_DIR, "best_discriminator.h5"))
                log_message(f"Best FID improved to {fid:.4f}. Models checkpointed.")
            else:
                wait += 1
                log_message(f"No improvement in FID for {wait} evaluations.")

            if config["early_stopping"]["enabled"] and wait >= config["early_stopping"]["patience"]:
                log_message("Early stopping triggered.")
                break

            save_samples(generator, epoch)
            generator.save_weights(os.path.join(CHECKPOINT_DIR, f"generator_epoch_{epoch}.h5"))
            discriminator.save_weights(os.path.join(CHECKPOINT_DIR, f"discriminator_epoch_{epoch}.h5"))
            log_message(f"💾 Saved weights and samples for epoch {epoch + 1}.")

        # TensorBoard logging
        with summary_writer.as_default():
            tf.summary.scalar("Generator Loss", g_loss, step=epoch)
            tf.summary.scalar("Discriminator Loss", d_loss, step=epoch)
            if fid is not None:
                tf.summary.scalar("FID Score", fid, step=epoch)
            tf.summary.scalar("Epoch Time (s)", time.time() - start_time, step=epoch)

    log_message("Training complete.")


# -------------------------------------
# SECTION 8: Generate Samples per Class (Conditional DCGAN)
# -------------------------------------
def generate_samples_per_class(generator, samples_per_class=1000, output_dir=SYNTHETIC_DIR):
    os.makedirs(output_dir, exist_ok=True)

    for class_idx in range(NUM_CLASSES):
        noise = np.random.normal(0, 1, (samples_per_class, LATENT_DIM))
        labels = np.full((samples_per_class, 1), class_idx)
        labels_onehot = tf.keras.utils.to_categorical(labels, NUM_CLASSES)

        gen_imgs = generator.predict([noise, labels_onehot], verbose=0)
        gen_imgs = (gen_imgs + 1) / 2.0  # Rescale from [-1, 1] to [0, 1]

        # Save to .npy files
        np.save(os.path.join(output_dir, f"gen_class_{class_idx}.npy"), gen_imgs)
        np.save(os.path.join(output_dir, f"labels_class_{class_idx}.npy"), labels)

        log_message(f"{samples_per_class} samples generated for class {class_idx} and saved to {output_dir}")


# -----------------------------------------
# SECTION 9: Build Classifier Architecture
# -----------------------------------------
def build_classifier(input_shape=(40, 40, 1), num_classes=9):
    model = models.Sequential(name="DCGAN_Evaluator")
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


# -----------------------------------------
# SECTION 10: Evaluate Classifier on Real vs Real+Synthetic
# -----------------------------------------
def evaluate_classifier_with_synthetic(generator):
    generate_samples_per_class(generator, samples_per_class=2000)

    # Load real test data
    test_data = np.load(os.path.join(DATA_PATH, "test_data.npy"), allow_pickle=True).reshape(-1, 40, 40, 1)
    test_data = (test_data - 0.5) * 2.0
    test_labels = np.load(os.path.join(DATA_PATH, "test_labels.npy"), allow_pickle=True)
    test_labels = tf.keras.utils.to_categorical(test_labels, NUM_CLASSES)

    # Load synthetic data
    X_gen, y_gen = load_synthetic_samples()

    # Combine with training data
    X_combined = np.concatenate([train_data, X_gen])
    y_combined = np.concatenate([train_labels, y_gen])

    # Train classifier with real + synthetic
    classifier = build_classifier()
    log_message("Training classifier on real + synthetic data...")
    classifier.fit(X_combined, y_combined, batch_size=256, epochs=100, validation_split=0.1)

    _, combined_acc = classifier.evaluate(test_data, test_labels, verbose=0)
    log_message(f"Classifier accuracy (Real + Synthetic): {combined_acc:.4f}")

    # Baseline with real only
    classifier_baseline = build_classifier()
    log_message("Training baseline classifier on real-only data...")
    classifier_baseline.fit(train_data, train_labels, batch_size=256, epochs=10, validation_split=0.1)

    _, baseline_acc = classifier_baseline.evaluate(test_data, test_labels, verbose=0)
    log_message(f"Classifier accuracy (Real only): {baseline_acc:.4f}")

    log_comparison_results(baseline_acc, combined_acc)


# -------------------------------------
# SECTION 11: Load Synthetic Samples (Conditional DCGAN)
# -------------------------------------
def load_synthetic_samples(directory=SYNTHETIC_DIR):
    gen_data = []
    gen_labels = []

    for class_idx in range(NUM_CLASSES):
        data_path = os.path.join(directory, f"gen_class_{class_idx}.npy")
        label_path = os.path.join(directory, f"labels_class_{class_idx}.npy")

        if not os.path.exists(data_path) or not os.path.exists(label_path):
            log_message(f"Missing synthetic files for class {class_idx}. Skipping.")
            continue

        x = np.load(data_path)
        y = np.load(label_path)

        gen_data.append(x)
        gen_labels.append(tf.keras.utils.to_categorical(y, NUM_CLASSES))

    if not gen_data:
        log_message("No synthetic data found!", display=True)
        return None, None

    X_gen = np.concatenate(gen_data)
    y_gen = np.concatenate(gen_labels)

    log_message(f"Loaded synthetic data: X_gen.shape={X_gen.shape}, y_gen.shape={y_gen.shape}")
    label_distribution = np.unique(np.argmax(y_gen, axis=1), return_counts=True)
    for label, count in zip(*label_distribution):
        log_message(f"Class {label}: {count} synthetic samples")

    return X_gen, y_gen


# -------------------------------------
# SECTION 12: Save Generator Outputs for Visual Preview
# -------------------------------------
def save_samples(generator, epoch, n=9):
    noise = np.random.normal(0, 1, (n, LATENT_DIM))
    labels = np.arange(n) % NUM_CLASSES
    labels_onehot = tf.keras.utils.to_categorical(labels, num_classes=NUM_CLASSES)

    generated_samples = generator.predict([noise, labels_onehot], verbose=0)

    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
    for i in range(n):
        axes[i].imshow(generated_samples[i].reshape(40, 40), cmap='gray')
        axes[i].set_title(f"Class {labels[i]}")
        axes[i].axis('off')
    plt.tight_layout()

    output_path = os.path.join(CHECKPOINT_DIR, f"cdgan_generated_epoch_{epoch}.png")
    plt.savefig(output_path)
    plt.close()

    log_message(f"Saved visual preview of generated samples to {output_path}")


# -------------------------------------
# SECTION 13: Preview Synthetic Samples from Disk
# -------------------------------------
def preview_synthetic_samples(directory=SYNTHETIC_DIR):
    plt.figure(figsize=(18, 2))

    for class_idx in range(NUM_CLASSES):
        filepath = os.path.join(directory, f"gen_class_{class_idx}.npy")
        if not os.path.exists(filepath):
            log_message(f"Missing file for class {class_idx}: {filepath}")
            continue

        samples = np.load(filepath)
        img = samples[0].reshape(40, 40)

        plt.subplot(1, NUM_CLASSES, class_idx + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Class {class_idx}")
        plt.axis('off')

    plt.tight_layout()
    preview_path = os.path.join(CHECKPOINT_DIR, "synthetic_preview_cdgan.png")
    plt.savefig(preview_path)
    plt.close()

    log_message(f"Saved preview of synthetic samples to: {preview_path}")


# -----------------------------------------
# SECTION 14: Log Comparison Results
# -----------------------------------------
def log_comparison_results(real_only_acc, real_plus_synth_acc):
    log_message("===== Classification Accuracy Summary =====")
    log_message(f"Real Only Accuracy:         {real_only_acc:.4f}")
    log_message(f"Real + Synthetic Accuracy:  {real_plus_synth_acc:.4f}")

    diff = real_plus_synth_acc - real_only_acc
    if diff > 0:
        log_message(f"Synthetic data improved accuracy by +{diff:.4f}")
    elif diff < 0:
        log_message(f"Accuracy dropped by {abs(diff):.4f} with synthetic data")
    else:
        log_message("No difference in performance")

    log_message("===========================================")


# -----------------------------------------
# SECTION 15: Main Runner Script
# -----------------------------------------
if __name__ == "__main__":
    if RUN_MODE == "train":
        log_message("Starting Conditional DCGAN Training...")
        train_conditional_dcgan()
        log_message("Training complete. Beginning evaluation...")

    # Always evaluate best generator
    generator = build_generator()
    generator.load_weights(os.path.join(CHECKPOINT_DIR, "best_generator.h5"))
    evaluate_classifier_with_synthetic(generator)

