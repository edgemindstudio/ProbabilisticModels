import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tqdm import trange
from tensorflow.keras.optimizers.legacy import Adam
from scipy.linalg import sqrtm
import datetime
import tensorflow_probability as tfp

#--------------------------------
# Configuration
#--------------------------------
IMG_SHAPE = (40, 40, 1)
LATENT_DIM = 100
BATCH_SIZE = 256
EPOCHS = 10000
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "USTC-TFC2016_malware")
os.makedirs("dcgan_checkpoints", exist_ok=True)

# -------------------------
# Load & Normalize Data [-1, 1] for tanh
# -------------------------
train_data = np.load(os.path.join(DATA_PATH, "train_data.npy"), allow_pickle=True)
print(train_data.ndim)
print("Train data shape before reshaping:", train_data.shape)
train_data = train_data.reshape(81000, 40, 40, 1)
# train_data = train_data.reshape(-1, *IMG_SHAPE)
train_data = (train_data - 0.5) * 2.0 # Change range from [0, 1] to [-1, 1] for tahn activation
print("Train data shape:", train_data.shape)
print(train_data.ndim)
print("--------------------------------------")

train_labels = np.load(os.path.join(DATA_PATH, "train_labels.npy"), allow_pickle=True)
train_labels = train_labels.reshape(-1)  # Ensure it’s 1D if needed
print("Train labels shape:", train_labels.shape)
print("Unique labels:", np.unique(train_labels))
print(train_labels.ndim)
print("---------------------------------------")

test_data = np.load(os.path.join(DATA_PATH, "test_data.npy"), allow_pickle=True)
test_data = test_data.reshape(-1, 40, 40, 1)
# test_data = test_data.reshape(-1, *IMG_SHAPE)
test_data = (test_data - 0.5) * 2.0
print("Test data shape:", test_data.shape)
print(test_data.ndim)
print("-------------------------------------")

test_labels = np.load(os.path.join(DATA_PATH, "test_labels.npy"), allow_pickle=True)
test_labels = test_labels.reshape(-1)
print("Test labels shape:", test_labels.shape)
print("Unique labels:", np.unique(test_labels))
print("--------------------------------------")

# Print data info
print("Data type:", train_data.dtype)
print("Pixel value range:", train_data.min(), "to", train_data.max())

# Preview a sample (flattened for printing)
print("Sample[0] reshaped (flattened):")
print(train_data[0].reshape(-1))  # Shows 1D version of the first image
print("------------------------------------------------------------------")
print(train_data[:2])  # Shows the first 2 images (each as a 40x40x1 array)
print(train_data[:2].shape)  # Output: (2, 40, 40, 1)
print("------------------------------------------------------------------")

# # View as image (if you want to visualize it with matplotlib)
# plt.imshow(train_data[0].reshape(40, 40), cmap='gray')
# plt.title("First Image in Training Data")
# plt.colorbar()
# plt.show()