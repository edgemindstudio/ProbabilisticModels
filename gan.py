import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, LeakyReLU, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import datetime

""" # Generator Model """
def build_generator():
    model = Sequential([
        Input(shape=(100,)),
        Dense(1024), # Increased from 256 to 1024 (Increased Capacity)
        BatchNormalization(),
        LeakyReLU(negative_slope=0.2),
        Dropout(0.2), # Prevent overfitting

        Dense(512),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.2),
        Dropout(0.2),

        Dense(256),
        BatchNormalization(),
        LeakyReLU(negative_slope=0.2),
        Dropout(0.2),

        Dense(1600, activation='sigmoid'),  # Output size 40x40 Changed from 'tanh' to 'sigmoid'
        Reshape((40, 40, 1))
    ])
    return model

""" # Discriminator Model """
def build_discriminator():
    model = Sequential([
        Input(shape=(40, 40, 1)),
        Flatten(),

        Dense(256), # Reduced from 512 to 256 to 384 to reduce Discriminator's Capacity
        LeakyReLU(negative_slope=0.2),
        Dropout(0.2), # Increased dropout back to 0.3 for stability

        Dense(64), # Reduced from 256 to 128 to 64 to reduce Discriminator's Capacity
        LeakyReLU(negative_slope=0.2),
        Dropout(0.3), # Increased dropout to help with generalization

        Dense(1, activation='sigmoid')
    ])
    return model

""" # Load dataset """
datafile_path = r"C:\Users\fonke\Downloads\USTC-TFC2016_malware"
train_data = np.load(datafile_path + "/train_data.npy", allow_pickle=True)
train_labels = np.load(datafile_path + "/train_labels.npy", allow_pickle=True)
test_data = np.load(datafile_path + "/test_data.npy", allow_pickle=True)
test_labels = np.load(datafile_path + "/test_labels.npy", allow_pickle=True)

# Reshape to ensure correct format for TensorFlow
train_data = train_data.reshape(-1, 40, 40, 1)  # Ensure shape is (samples, 40, 40, 1)
test_data = test_data.reshape(-1, 40, 40, 1)

# Normalize dataset to match sigmoid output
train_data = train_data / 255.0
test_data = test_data / 255.0

# print("Train Data Shape:", train_data.shape)
# print("Train Label Shape:", train_labels.shape)
# print("Test Data Shape:", test_data.shape)
# print("Test Label Shape:", test_labels.shape)

""" # Initialize models """
generator = build_generator()
discriminator = build_discriminator()

# 👇 Must set trainable before compiling!
discriminator.trainable = True # Ensure Discriminator is trainable before standalone training

# Compile discriminator
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(0.00002, 0.5),
                      metrics=['accuracy']) # Slower learning rate (Changed learning rate from 0.0002 to 0.0001 to 0.00005)

# 👇 Freeze D before compiling the GAN
discriminator.trainable = False  # Freeze Discriminator when training the Generator

# Build GAN
z = tf.keras.Input(shape=(100,))
generated_image = generator(z)
validity = discriminator(generated_image)

gan = tf.keras.Model(z, validity)
gan.compile(loss='binary_crossentropy',
            optimizer=Adam(0.0003, 0.5)) # Increase generator learning rate from 0.0002 to 0.0003

""" # Helper: Generate synthetic samples """
def generate_synthetic_samples(generator, num_samples=10):
    """Generates malware traffic samples using the trained Generator."""
    noise = np.random.normal(0, 1, (num_samples, 100))  # Random noise input
    generated_samples = generator.predict(noise)  # Generate fake samples

    # Rescale from [-1, 1] to [0, 255] for better visualization
    generated_samples = ((generated_samples + 1) * 127.5).astype(np.uint8)

    return generated_samples

""" # Helper: Evaluate Discriminator """
def evaluate_discriminator(discriminator, real_data, generator, num_samples=100):
    """Evaluates the Discriminator's performance."""
    noise = np.random.normal(0, 1, (num_samples, 100))
    fake_data = generator.predict(noise)

    real_labels = np.ones((num_samples, 1)) * 0.85 # Increase Label Smoothing
    fake_labels = np.zeros((num_samples, 1)) + 0.1 # Increase Fake Label Smoothing

    real_loss, real_acc = discriminator.evaluate(real_data[:num_samples], real_labels, verbose=0)
    fake_loss, fake_acc = discriminator.evaluate(fake_data, fake_labels, verbose=0)

    print(f"Discriminator Accuracy on Real Data: {real_acc * 100:.2f}%")
    print(f"Discriminator Accuracy on Fake Data: {fake_acc * 100:.2f}%")

    return real_acc, fake_acc

""" # Helper: Visualize Generator Output """
def visualize_generated_samples(generator, num_samples=10):
    """Displays generated malware traffic images."""
    generated_samples = generate_synthetic_samples(generator, num_samples)

    fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
    for i in range(num_samples):
        axes[i].imshow(generated_samples[i].reshape(40, 40), cmap='gray')
        axes[i].axis('off')
    plt.show()

""" # Logging function """
def log_results(epoch, d_loss, g_loss, real_acc, fake_acc, log_file="gan_training_log.txt"):
    """Logs training results to a file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (f"[{timestamp}] Epoch {epoch}: D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}, "
                 f"Real Acc: {real_acc:.2f}%, Fake Acc: {fake_acc:.2f}%\n")

    print(log_entry.strip())  # Print to console

    # Append to log file
    with open(log_file, "a") as f:
        f.write(log_entry)


""" # Training loop """
def train_gan(epochs=10000, batch_size=256): # Increase batch size from 64 to 128 to 256
    log_file = "gan_training_log.txt"

    for epoch in range(epochs):
        # Train Discriminator / Sample and slightly perturb real images
        idx = np.random.randint(0, train_data.shape[0], batch_size)
        real_images = train_data[idx] + np.random.normal(0, 0.02, train_data[idx].shape)  # Add noise

        # Generate Fake Images
        noise = np.random.normal(0, 1, (batch_size, 100))
        fake_images = generator.predict(noise)
        # Add slight noise to real_images and fake_images for regularization
        fake_images += np.random.normal(0, 0.02, fake_images.shape)

        # Smoothing Labels
        real_labels = np.ones((batch_size, 1)) * 0.85 # Label smoothing (Real)
        fake_labels = np.zeros((batch_size, 1)) + 0.1 # One-sided label smoothing (Fake)

        # ✅ Set discriminator trainable before training
        discriminator.trainable = True  # Ensure Discriminator is trainable
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ✅ Freeze discriminator before training generator
        discriminator.trainable = False

        # Train Generator
        noise = np.random.normal(0, 1, (batch_size, 100))
        misleading_labels = np.ones((batch_size, 1)) * 0.85
        #discriminator.trainable = False  # Freeze Discriminator while training Generator
        g_loss = gan.train_on_batch(noise, misleading_labels)

        # Log and evaluate
        if epoch % 1000 == 0:
            real_acc, fake_acc = evaluate_discriminator(discriminator, train_data, generator)
            print(f"[Epoch {epoch}] - D Loss: {d_loss[0]:.4f}, G Loss: {g_loss:.4f}, "
                  f"Real Acc: {real_acc*100:.2f}%, Fake Acc: {fake_acc*100:.2f}%")
            log_results(epoch, d_loss[0], g_loss, real_acc * 100, fake_acc * 100, log_file)
            # Save generated samples
            visualize_generated_samples(generator, num_samples=5)

""" # Start Training """
train_gan(epochs=10000, batch_size=256)

""" # Evaluate the trained model """
visualize_generated_samples(generator, num_samples=10)
evaluate_discriminator(discriminator, train_data, generator, num_samples=500)