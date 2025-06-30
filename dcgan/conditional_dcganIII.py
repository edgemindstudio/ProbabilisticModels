import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt
from tqdm import trange
from scipy.linalg import sqrtm
import tensorflow_probability as tfp
from datetime import datetime
import sys

LOG_FILE = "conditional_dcgan_result.txt"

def log_message(message, display=True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] {message}"
    if display:
        print(full_msg)
    with open(LOG_FILE, "a") as f:
        f.write(full_msg + "\n")

# -------------------------------
# Config (Hyperparameters)
# -------------------------------
IMG_SHAPE = (40, 40, 1)
LATENT_DIM = 100
NUM_CLASSES = 9
BATCH_SIZE = 256
EPOCHS = 5000
LR = 0.0002
BETA_1 = 0.5

# Locate Data Folder
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "USTC-TFC2016_malware")

# Log Files
LOG_DIR = "logs"
TENSORBOARD_DIR = os.path.join(LOG_DIR, "tensorboard", datetime.now().strftime("%Y%m%d-%H%M%S"))
# TensorBoard writer
summary_writer = tf.summary.create_file_writer(TENSORBOARD_DIR)

os.makedirs("cdgan_checkpoints", exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Save hyperparameters
with open(os.path.join(LOG_DIR, "hyperparameters.txt"), "w") as f:
    f.write(f"IMG_SHAPE={IMG_SHAPE}, LATENT_DIM={LATENT_DIM}, NUM_CLASSES={NUM_CLASSES}, \
            BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}, LR={LR}, BETA_1={BETA_1}\n")

# -------------------------------
# Load & Normalize Data
# -------------------------------
train_data = np.load(os.path.join(DATA_PATH, "train_data.npy"), allow_pickle=True)
train_labels = np.load(os.path.join(DATA_PATH, "train_labels.npy"), allow_pickle=True)
train_data = train_data.reshape(-1, *IMG_SHAPE)
train_data = (train_data - 0.5) * 2.0
train_labels = tf.keras.utils.to_categorical(train_labels, NUM_CLASSES)

# -------------------------------
# FID Calculation
# -------------------------------
inception_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    pooling='avg', input_shape=(75, 75, 3))

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
    covmean = sqrtm(sigma1 @ sigma2) # OR covmean = sqrtm(np.matmul(sigma1, sigma2)) matmul == @ from python 3.5
    if np.iscomplexobj(covmean): covmean = covmean.real
    if np.isnan(covmean).any(): return 1e6
    return (ssdiff + tf.linalg.trace(sigma1 + sigma2 - 2 * covmean)).numpy()


# -------------------------------
# Generator Model
# -------------------------------
def build_generator():
    noise_input = layers.Input(shape=(LATENT_DIM,))
    label_input = layers.Input(shape=(NUM_CLASSES,))

    # Dense project + reshape noise
    x = layers.Dense(5 * 5 * 256, use_bias=False)(noise_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Reshape((5, 5, 256))(x)

    # Project label to same spatial size and depth-1
    label_embedding = layers.Dense(5 * 5 * 1, use_bias=False)(label_input)
    label_embedding = layers.Reshape((5, 5, 1))(label_embedding)

    # Concatenate label embedding as a separate channel
    x = layers.Concatenate()([x, label_embedding])

    # 4. Upsample + Conv (1)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(128, kernel_size=3, padding='same', use_bias=False, name="gen_conv1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # 5. Upsample + Conv (2)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(64, kernel_size=3, padding='same', use_bias=False, name="gen_conv2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # 6. Upsample + Conv (3) → output layer
    x = layers.UpSampling2D()(x)
    out = layers.Conv2D(1, kernel_size=3, padding='same', activation='tanh', use_bias=False, name="gen_output")(x)

    return models.Model(inputs=[noise_input, label_input], outputs=out, name="Conditional_Generator")


# -------------------------------
# Discriminator Model
# -------------------------------
def build_discriminator():
    image_input = layers.Input(shape=IMG_SHAPE)
    label_input = layers.Input(shape=(NUM_CLASSES,))

    # Project and reshape label
    label_map = layers.Dense(np.prod(IMG_SHAPE))(label_input)
    label_map = layers.Reshape(IMG_SHAPE)(label_map)

    # Concatenate along channel axis
    merged = layers.Concatenate(axis=-1)([image_input, label_map])  # Shape: (40, 40, 2)

    # Conv layers
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same', name="disc_conv1")(merged)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same', name="disc_conv2")(x)
    x = layers.BatchNormalization()(x)  # Add BN here
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    return models.Model(inputs=[image_input, label_input],
                        outputs=out, name="Conditional_Discriminator")

# -------------------------------
# Sample Generating Function
# -------------------------------
def generate_samples_per_class(generator, samples_per_class=1000, output_dir="synthetic_samples"):
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

        log_message(f"{samples_per_class} samples generated for class {class_idx} and saved to {output_dir}.")

# -------------------------------
# Simple CNN Classifier
# -------------------------------
def build_classifier(input_shape=(40, 40, 1), num_classes=9):
    model = models.Sequential(name="Malware_Classifier")
    model.add(layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# --------------------------------------------------
# Train & Evaluate Real-Only Baseline Classifier
# --------------------------------------------------
def train_classifier_on_real_only():
    log_message("Training classifier on real data only...")

    classifier = build_classifier()
    classifier.fit(train_data, train_labels, batch_size=256, epochs=10, validation_split=0.1)

    _, real_only_acc = classifier.evaluate(test_data, test_labels, verbose=0)
    log_message(f"[BASELINE] Classifier accuracy (Real only): {real_only_acc:.4f}")

    return real_only_acc

# -------------------------------
# Load Synthetic Data
# -------------------------------
def load_synthetic_samples(directory="synthetic_samples"):
    gen_data = []
    gen_labels = []

    for class_idx in range(NUM_CLASSES):
        x = np.load(os.path.join(directory, f"gen_class_{class_idx}.npy"))
        y = np.load(os.path.join(directory, f"labels_class_{class_idx}.npy"))
        gen_data.append(x)
        gen_labels.append(tf.keras.utils.to_categorical(y, NUM_CLASSES))

    X_gen = np.concatenate(gen_data)
    y_gen = np.concatenate(gen_labels)

    # Log shapes and label distribution
    log_message(f"Loaded synthetic data: X_gen.shape={X_gen.shape}, y_gen.shape={y_gen.shape}")
    label_distribution = np.unique(np.argmax(y_gen, axis=1), return_counts=True)
    for label, count in zip(*label_distribution):
        log_message(f"Class {label}: {count} samples")

    return X_gen, y_gen

# -------------------------
# Save Generator Outputs
# -------------------------
def save_samples(generator, epoch, n=10):
    noise = np.random.normal(0, 1, (n, 100))
    labels = np.arange(n) % 9  # Generate labels for classes 0 to 8
    labels = labels.reshape(-1, 1)

    labels = tf.keras.utils.to_categorical(labels, num_classes=9)
    generated_samples = generator.predict([noise, labels], verbose=0)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
    for i in range(n):
        axes[i].imshow(generated_samples[i].reshape(40, 40), cmap='gray')
        axes[i].set_title(f"Class {labels[i][0]}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(f"cdgan_generated_epoch_{epoch}.png")
    plt.close()

# ----------------------------------------
# Visual Check for One Sample per Class
# ----------------------------------------
def preview_synthetic_samples(directory="synthetic_samples"):
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
    plt.savefig("synthetic_preview.png")
    plt.close()


# -----------------------------------------
# Training Loop with TensorBoard Logging
# -----------------------------------------
def train_cdgan():
    log_message("Initializing Conditional Generator...")
    generator = build_generator()
    log_message("Generator initialized.")

    log_message("Initializing Conditional Discriminator...")
    discriminator = build_discriminator()
    log_message("Discriminator initialized.")

    d_optimizer = Adam(LR, BETA_1)
    g_optimizer = Adam(LR, BETA_1)

    discriminator.compile(optimizer=d_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    noise_input = layers.Input(shape=(LATENT_DIM,))
    label_input = layers.Input(shape=(NUM_CLASSES,))
    gen_img = generator([noise_input, label_input])
    discriminator.trainable = False
    validity = discriminator([gen_img, label_input])
    combined = models.Model([noise_input, label_input], validity)
    combined.compile(optimizer=g_optimizer, loss='binary_crossentropy')

    for epoch in trange(EPOCHS):
        log_message(f"Epoch {epoch + 1}/{EPOCHS} starting...")
        start_time = time.time()

        idx = np.random.randint(0, train_data.shape[0], BATCH_SIZE)
        real_imgs = train_data[idx]
        real_lbls = train_labels[idx]

        # === Generator input ===
        # noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
        noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))

        # === Use random labels for fake images ===
        fake_labels = np.random.randint(0, NUM_CLASSES, BATCH_SIZE).reshape(-1, 1)
        fake_labels_onehot = tf.keras.utils.to_categorical(fake_labels, NUM_CLASSES)

        # === Generate fake images ===
        # gen_imgs = generator.predict([noise, real_lbls], verbose=0)
        gen_imgs = generator.predict([noise, fake_labels_onehot], verbose=0)

        # # Now apply noise injection | Add small Gaussian noise
        # real_imgs += np.random.normal(0, 0.01, real_imgs.shape)  # Real image noise
        # gen_imgs += np.random.normal(0, 0.01, gen_imgs.shape)  # Fake image noise
        if epoch > 200:
            real_imgs += np.random.normal(0, 0.01, real_imgs.shape)
            gen_imgs += np.random.normal(0, 0.01, gen_imgs.shape)

        # Smooth labels
        real_y = np.random.uniform(0.9, 1.0, size=(BATCH_SIZE, 1))  # Label smoothing for real
        fake_y = np.random.uniform(0.0, 0.1, size=(BATCH_SIZE, 1))  # Noisy labels for fake

        # === Train Discriminator ===
        discriminator.trainable = True
        d_loss_real = discriminator.train_on_batch([real_imgs, real_lbls], real_y)
        # d_loss_fake = discriminator.train_on_batch([gen_imgs, real_lbls], fake_y)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, fake_labels_onehot], fake_y)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # === Train Generator ===
        discriminator.trainable = False
        misleading_y = np.ones((BATCH_SIZE, 1))  # <== This was missing
        # g_loss = combined.train_on_batch([noise, real_lbls], misleading_y)
        g_loss = combined.train_on_batch([noise, fake_labels_onehot], misleading_y)

        fid = None

        if (epoch + 1) % 100 == 0:
            # Example logging (use your actual variable names):
            log_message(f"Generator Loss: {g_loss:.4f}")
            log_message(f"Discriminator Loss: {d_loss[0]:.4f}")
            log_message(f"Discriminator Accuracy: {d_loss[1] * 100:.2f}%")

            log_message(f"Discriminator Real Accuracy: {d_loss_real[1] * 100:.2f}%")
            log_message(f"Discriminator Fake Accuracy: {d_loss_fake[1] * 100:.2f}%")

            fid = calculate_fid(real_imgs[:100], gen_imgs[:100])
            log_message(f"FID Score at Epoch {epoch + 1}: {fid:.4f}")

            save_samples(generator, epoch)
            log_message(f"Generated samples saved at epoch {epoch + 1}.")

            # Save weights
            generator.save_weights(f"cdgan_checkpoints/generator_epoch_{epoch}.h5")
            discriminator.save_weights(f"cdgan_checkpoints/discriminator_epoch_{epoch}.h5")
            log_message(f"Model weights saved at epoch {epoch + 1}.")

        # TensorBoard logging
        with summary_writer.as_default():
            tf.summary.scalar("Generator Loss", g_loss, step=epoch)
            tf.summary.scalar("Discriminator Loss", d_loss[0], step=epoch)
            tf.summary.scalar("Discriminator Accuracy", d_loss[1], step=epoch)
            if fid is not None:
                tf.summary.scalar("FID Score", fid, step=epoch)
            tf.summary.scalar("Epoch Time (s)", time.time() - start_time, step=epoch)

    log_message("Training complete.")

# -------------------------------
# Result Logger
# -------------------------------
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

# -------------------------------
# Train Classifier
# -------------------------------
# Build generator globally
generator = build_generator()

if __name__ == "__main__":
    train_cdgan()
    generate_samples_per_class(generator, samples_per_class=2000)

    # Load test data
    test_data = np.load(os.path.join(DATA_PATH, "test_data.npy"), allow_pickle=True).reshape(-1, 40, 40, 1)
    test_data = (test_data - 0.5) * 2.0
    test_labels = np.load(os.path.join(DATA_PATH, "test_labels.npy"), allow_pickle=True)
    test_labels = tf.keras.utils.to_categorical(test_labels, NUM_CLASSES)

    # Load synthetic data
    X_gen, y_gen = load_synthetic_samples()

    # Combine real + generated
    X_combined = np.concatenate([train_data, X_gen])
    y_combined = np.concatenate([train_labels, y_gen])

    # Train classifier on real + synthetic data
    classifier = build_classifier()
    log_message("Training classifier on real + synthetic data...")
    classifier.fit(X_combined, y_combined, batch_size=256, epochs=100, validation_split=0.1)

    _, real_plus_synth_acc = classifier.evaluate(test_data, test_labels, verbose=0)
    log_message(f"Classifier accuracy on test set (Real + Synthetic): {real_plus_synth_acc:.4f}")

    real_only_acc = train_classifier_on_real_only()

    # Diagnostic sample visualization
    preview_synthetic_samples()

    # Log final comparison
    log_comparison_results(real_only_acc, real_plus_synth_acc)
