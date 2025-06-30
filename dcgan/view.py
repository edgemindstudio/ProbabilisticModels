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

from conditional_dcganIII import build_generator, build_discriminator

generator = build_generator()
generator.load_weights("cdgan_checkpoints/generator_epoch_1999.h5")

discriminator = build_discriminator()
discriminator.load_weights("cdgan_checkpoints/discriminator_epoch_1999.h5")

noise = np.random.normal(0, 1, (9, 100))
labels = tf.keras.utils.to_categorical(np.arange(9), num_classes=9)
generated_images = generator.predict([noise, labels], verbose=0)


plt.figure(figsize=(12, 2))
for i in range(9):
    plt.subplot(1, 9, i + 1)
    plt.imshow(generated_images[i].reshape(40, 40), cmap="gray")
    plt.title(f"Class {i}")
    plt.axis('off')
plt.tight_layout()
plt.savefig("preview_loaded_generator.png")
plt.show()

generator.summary()
print("-----------------------------------")
print("-----------------------------------")

discriminator.summary()

#