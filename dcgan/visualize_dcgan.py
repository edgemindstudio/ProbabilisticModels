import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from conditional_dcganIII import build_generator

# Load the trained generator model
generator = build_generator()
generator.load_weights("cdgan_checkpoints/best_generator.h5")

# Generate one sample per class (0 through 8)
LATENT_DIM = 100
NUM_CLASSES = 9

noise = np.random.normal(0, 1, (NUM_CLASSES, LATENT_DIM))
labels = np.arange(NUM_CLASSES).reshape(-1, 1)
labels_onehot = to_categorical(labels, num_classes=NUM_CLASSES)

generated_images = generator.predict([noise, labels_onehot], verbose=0)

# Plot the generated images
plt.figure(figsize=(18, 2))
for i in range(NUM_CLASSES):
    plt.subplot(1, NUM_CLASSES, i + 1)
    plt.imshow(generated_images[i].reshape(40, 40), cmap='gray')
    plt.title(f"Class {i}")
    plt.axis('off')

plt.tight_layout()
plt.savefig("conditional_dcgan_preview_best_generator.png")
plt.show()