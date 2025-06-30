# =====================================
# SECTION 1: Imports and Initial Setup (PyTorch Version)
# =====================================

import os
import yaml
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from scipy.linalg import sqrtm
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torchvision.models import inception_v3
from torchvision import transforms
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T

# ==============================
# Set Random Seeds
# ==============================
def set_random_seeds(seed=42):
    """Ensure reproducibility across NumPy, PyTorch, and random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

# ==============================
# Prepare PyTorch Dataset Utility
# ==============================
def prepare_torch_dataset(x, y, batch_size=128, shuffle=True):
    """Converts NumPy arrays into a batched PyTorch DataLoader."""
    tensor_x = torch.tensor(x, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)



# =====================================
# SECTION 2A: Data Loading and Preprocessing
# =====================================
def load_malware_dataset(data_path, img_shape, num_classes, val_fraction=0.5):
    """
    Load and preprocess malware dataset for Masked Autoregressive Flow.

    Args:
        data_path (str): Path to dataset .npy files
        img_shape (tuple): Image shape (H, W, C)
        num_classes (int): Number of classes
        val_fraction (float): Fraction of test data to use as validation

    Returns:
        Tuple: (x_train, y_train), (x_val, y_val), (x_test, y_test) as PyTorch tensors
    """
    x_train = np.load(os.path.join(data_path, "train_data.npy")).astype(np.float32) / 255.0
    y_train = np.load(os.path.join(data_path, "train_labels.npy")).astype(np.int64)
    x_test = np.load(os.path.join(data_path, "test_data.npy")).astype(np.float32) / 255.0
    y_test = np.load(os.path.join(data_path, "test_labels.npy")).astype(np.int64)

    # Flatten images for MAF
    x_train = x_train.reshape((-1, np.prod(img_shape)))
    x_test = x_test.reshape((-1, np.prod(img_shape)))

    # One-hot encode labels if needed later
    y_train_onehot = np.eye(num_classes)[y_train]
    y_test_onehot = np.eye(num_classes)[y_test]

    # Split test set into validation and test sets
    split_idx = int(len(x_test) * val_fraction)
    x_val, y_val = x_test[:split_idx], y_test[:split_idx]
    x_test, y_test = x_test[split_idx:], y_test[split_idx:]

    return (
        (torch.from_numpy(x_train), torch.from_numpy(y_train)),
        (torch.from_numpy(x_val), torch.from_numpy(y_val)),
        (torch.from_numpy(x_test), torch.from_numpy(y_test))
    )

def get_data_loaders(x_train, y_train, x_val, y_val, x_test, y_test, batch_size):
    """
    Create PyTorch DataLoaders for training, validation, and testing.

    Args:
        x_train, y_train, x_val, y_val, x_test, y_test (Tensor): Data tensors
        batch_size (int): Batch size for DataLoaders

    Returns:
        Tuple: train_loader, val_loader, test_loader
    """
    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)
    test_ds = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader



# =====================================
# SECTION 2B: Evaluation Metric Placeholders
# =====================================

# ------------------------------
# Metric 1: Fréchet Inception Distance (FID)
# ------------------------------
def calculate_fid(real_images, generated_images, device="cpu"):
    """
    Compute Fréchet Inception Distance (FID) between real and generated samples.

    Args:
        real_images (np.ndarray): Real images in [0,1], shape [N, H, W, C]
        generated_images (np.ndarray): Generated images in [0,1], shape [N, H, W, C]
        device (str): 'cuda' or 'cpu'

    Returns:
        float: FID score
    """
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # grayscale to RGB if needed
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def get_activations(images):
        images = torch.from_numpy(images).permute(0, 3, 1, 2).float().to(device)
        images = images / 255.0 if images.max() > 1.0 else images
        activations = []
        with torch.no_grad():
            for img in images:
                img = preprocess(img).unsqueeze(0)
                act = model(img)[0].cpu().numpy()
                activations.append(act)
        return np.array(activations)

    act1 = get_activations(real_images)
    act2 = get_activations(generated_images)

    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean))


# ------------------------------
# Metric 2: Jensen-Shannon and KL Divergence
# ------------------------------
def js_divergence(p, q):
    """
    Compute Jensen-Shannon divergence between two distributions.

    Args:
        p, q (np.ndarray): Probability distributions

    Returns:
        float: JS divergence
    """
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def kl_divergence(p, q):
    """
    Compute Kullback-Leibler divergence between two distributions.

    Args:
        p, q (np.ndarray): Probability distributions

    Returns:
        float: KL divergence
    """
    return entropy(p, q)


# ------------------------------
# Metric 3: Classifier Utility Metrics
# ------------------------------
def evaluate_classifier_metrics(y_true, y_pred, average="macro"):
    """
    Evaluate classification performance using standard metrics.

    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels

    Returns:
        dict: Dictionary containing accuracy, precision, recall, F1, and confusion matrix
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
    Compute diversity score using feature-wise variance.

    Args:
        samples (np.ndarray): Generated samples [N, D]

    Returns:
        float: Diversity score (mean feature-wise variance)
    """
    return float(np.mean(np.var(samples.reshape(samples.shape[0], -1), axis=0)))



# =====================================
# SECTION 3: Masked Autoregressive Flow (MAF) Model Definition
# =====================================

# ==============================
# Masked Linear Layer (MADE building block)
# ==============================
class MaskedLinear(nn.Linear):
    """
    Linear layer with autoregressive mask for MADE and MAF.

    Args:
        in_features (int): Input dimensionality
        out_features (int): Output dimensionality
        mask (torch.Tensor): Binary mask tensor [out_features, in_features]
    """

    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        """
        Set the autoregressive mask.

        Args:
            mask (torch.Tensor): Mask tensor of shape [out_features, in_features]
        """
        self.mask.data.copy_(mask)

    def forward(self, input):
        return F.linear(input, self.weight * self.mask, self.bias)

# ==============================
# MADE Block
# ==============================
class MADE(nn.Module):
    """
    Masked Autoencoder for Distribution Estimation (MADE).
    Used inside MAF as the core flow block.

    Args:
        input_dim (int): Dimensionality of input
        hidden_dims (list): Hidden layer sizes
    """

    def __init__(self, input_dim, hidden_dims=[128, 128]):
        super(MADE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Degree assignments for autoregressive masks
        self.degrees = []

        # Layers
        layers = []
        hs = [input_dim] + hidden_dims + [input_dim * 2]
        for h0, h1 in zip(hs[:-1], hs[1:]):
            layers.append(MaskedLinear(h0, h1))
            layers.append(nn.ReLU())
        layers.pop()  # remove last ReLU
        self.net = nn.Sequential(*layers)

        self.create_masks()

    def create_masks(self):
        """
        Create masks for all MaskedLinear layers to enforce autoregressive property.
        """
        L = len(self.hidden_dims)
        degrees = []

        # Input degrees
        degrees.append(torch.arange(1, self.input_dim + 1))

        # Hidden layer degrees
        for h in self.hidden_dims:
            degrees.append(torch.randint(1, self.input_dim + 1, (h,)))

        # Output degrees (for mu and log_sigma)
        degrees.append(torch.arange(1, self.input_dim + 1).repeat(2))

        # Apply masks
        layers = [layer for layer in self.net if isinstance(layer, MaskedLinear)]
        for i, layer in enumerate(layers):
            d_in = degrees[i]
            d_out = degrees[i + 1]
            mask = (d_out[:, None] >= d_in[None, :]).float()
            layer.set_mask(mask)

    def forward(self, x):
        """
        Forward pass returning mu and log_sigma.

        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]

        Returns:
            mu (torch.Tensor), log_sigma (torch.Tensor)
        """
        output = self.net(x)
        mu, log_sigma = output.chunk(2, dim=1)
        return mu, log_sigma

# ==============================
# MAF Model
# ==============================
class MAF(nn.Module):
    """
    Masked Autoregressive Flow Model composed of stacked MADE blocks.

    Args:
        input_dim (int): Dimensionality of input
        num_flows (int): Number of MADE layers to stack
        hidden_dims (list): Hidden layer sizes in MADE
    """

    def __init__(self, input_dim, num_flows=5, hidden_dims=[128, 128]):
        super(MAF, self).__init__()
        self.input_dim = input_dim
        self.flows = nn.ModuleList([
            MADE(input_dim, hidden_dims) for _ in range(num_flows)
        ])

    def forward(self, x):
        """
        Forward transformation through the MAF flows.

        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]

        Returns:
            z (torch.Tensor): Latent variable
            log_det (torch.Tensor): Log-determinant of Jacobian
        """
        log_det = torch.zeros(x.size(0), device=x.device)
        z = x
        for flow in self.flows:
            mu, log_sigma = flow(z)
            z = (z - mu) * torch.exp(-log_sigma)
            log_det += -log_sigma.sum(dim=1)
        return z, log_det

    def log_prob(self, x):
        """
        Compute log-likelihood of input under the MAF model.

        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]

        Returns:
            log_prob (torch.Tensor): Log-likelihood [batch_size]
        """
        z, log_det = self.forward(x)
        log_base = -0.5 * ((z ** 2) + np.log(2 * np.pi)).sum(dim=1)
        return log_base + log_det

    def inverse(self, z, num_iter=1):
        """
        Inverse transformation: sample x given z using autoregressive inversion.

        Args:
            z (torch.Tensor): Latent variable [batch_size, input_dim]
            num_iter (int): Number of fixed-point iterations (usually 1 is sufficient)

        Returns:
            x (torch.Tensor): Reconstructed data sample [batch_size, input_dim]
        """
        x = z.clone()
        for flow in reversed(self.flows):
            for _ in range(num_iter):
                mu, log_sigma = flow(x)
                x = z * torch.exp(log_sigma) + mu
        return x

    def sample(self, num_samples):
        """
        Sample data points from the MAF model by inverting the flows.

        Args:
            num_samples (int): Number of samples to generate

        Returns:
            samples (torch.Tensor): Generated samples [num_samples, input_dim]
        """
        z = torch.randn(num_samples, self.input_dim)
        for flow in reversed(self.flows):
            mu, log_sigma = flow(z)
            z = z * torch.exp(log_sigma) + mu
        return z

def build_maf_model(input_dim, num_layers=5, hidden_dims=[128, 128]):
    """
    Build Masked Autoregressive Flow (MAF) model.

    Args:
        input_dim (int): Flattened image dimension
        num_layers (int): Number of flow layers (MADE blocks)
        hidden_dims (list): Hidden dimensions for MADE

    Returns:
        MAF: PyTorch MAF model
    """
    return MAF(input_dim=input_dim, num_flows=num_layers, hidden_dims=hidden_dims)





# =====================================
# SECTION 4: MAF Training Loop with Early Stopping
# =====================================

def train_maf_model(
    model,
    train_loader,
    val_loader,
    config,
    writer,
    checkpoint_dir,
    device=torch.device("cpu")
):
    """
    Train the Masked Autoregressive Flow (MAF) model with early stopping.

    Args:
        model (MAF): Initialized MAF model
        train_loader (DataLoader): PyTorch DataLoader for training data
        val_loader (DataLoader): PyTorch DataLoader for validation data
        config (dict): Training configurations
        writer (SummaryWriter): TensorBoard summary writer
        checkpoint_dir (str): Directory to store best model checkpoint
        device (torch.device): Device to run training on

    Returns:
        model (MAF): Trained model with best weights
    """
    epochs = config.get("EPOCHS", 100)
    lr = config.get("LR", 1e-3)
    patience = config.get("PATIENCE", 10)
    clip_grad = config.get("CLIP_GRAD", 1.0)

    optimizer = Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    patience_counter = 0

    os.makedirs(checkpoint_dir, exist_ok=True)

    model = model.to(device)

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # -----------------------------
        # Training Loop
        # -----------------------------
        model.train()
        train_losses = []
        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            neg_log_likelihood = -model.log_prob(x_batch).mean()
            neg_log_likelihood.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            train_losses.append(neg_log_likelihood.item())

        avg_train_loss = np.mean(train_losses)

        # -----------------------------
        # Validation Loop
        # -----------------------------
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, _ in val_loader:
                x_batch = x_batch.to(device)
                val_nll = -model.log_prob(x_batch).mean()
                val_losses.append(val_nll.item())

        avg_val_loss = np.mean(val_losses)

        # -----------------------------
        # TensorBoard Logging
        # -----------------------------
        writer.add_scalar("Loss/Train_NLL", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val_NLL", avg_val_loss, epoch)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:03d} | Train NLL: {avg_train_loss:.4f} | Val NLL: {avg_val_loss:.4f} | Time: {elapsed:.2f}s"
        )

        # -----------------------------
        # Early Stopping and Checkpointing
        # -----------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            checkpoint_path = os.path.join(
                checkpoint_dir, f"maf_best_epoch_{epoch:03d}.pt"
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to: {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return model



# =====================================
# SECTION 5: Synthetic Sampling and Saving Outputs
# =====================================

def sample_from_maf(model, num_samples, input_dim, device=torch.device("cpu")):
    """
    Generate synthetic samples from a trained MAF model.

    Args:
        model (MAF): Trained MAF model
        num_samples (int): Number of samples to generate
        input_dim (int): Dimensionality of each sample
        device (torch.device): Device to run sampling on

    Returns:
        np.ndarray: Synthetic samples [num_samples, input_dim] in [0, 1]
    """
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, input_dim, device=device)
        samples = model.inverse(z).cpu().numpy()
        samples = np.clip(samples, 0.0, 1.0)
    return samples


def save_synthetic_samples(samples, save_dir, prefix="maf"):
    """
    Save generated synthetic samples as .npy files.

    Args:
        samples (np.ndarray): Generated samples [N, D] or [N, H, W, C]
        save_dir (str): Directory to save samples
        prefix (str): Prefix for filenames
    """
    os.makedirs(save_dir, exist_ok=True)
    for idx, sample in enumerate(samples):
        np.save(os.path.join(save_dir, f"{prefix}_{idx}.npy"), sample)


def visualize_samples(samples, img_shape, num_display=16, save_path="maf_samples.png"):
    """
    Display a grid of generated samples.

    Args:
        samples (np.ndarray): Generated samples [N, D] or [N, H, W, C]
        img_shape (tuple): Image shape (H, W, C)
        num_display (int): Number of samples to display
    """
    samples = samples.reshape(-1, *img_shape)
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
    plt.savefig(save_path, dpi=300)
    plt.close()



# =====================================
# SECTION 6: Evaluation Using Synthetic Samples
# =====================================

def compute_sample_diversity(samples):
    """
    Compute diversity score using feature variance.

    Args:
        samples (Tensor or ndarray): Generated samples [N, D]

    Returns:
        float: Mean feature-wise variance
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu().numpy()
    return float(np.mean(np.var(samples.reshape(samples.shape[0], -1), axis=0)))

def evaluate_generation_quality(real_images, generated_images, device='cpu', alpha=1e-6):
    """
    Full evaluation pipeline for comparing generated and real images.

    Args:
        real_images (Tensor): Real images [N, C, H, W] in [0, 1]
        generated_images (Tensor): Generated images [N, C, H, W] in [0, 1]
        device (str): 'cpu' or 'cuda'
        alpha (float): Smoothing factor for histogram-based metrics

    Returns:
        dict: Dictionary containing FID, JS, KL, and Diversity Score
    """
    # Ensure tensors on CPU for numpy operations
    if isinstance(real_images, torch.Tensor):
        real_images = real_images.cpu().numpy()
    if isinstance(generated_images, torch.Tensor):
        generated_images = generated_images.cpu().numpy()

    # Metric 1: FID Score
    fid = calculate_fid(real_images, generated_images, device=device)

    # Metric 2: Histogram-based JS and KL Divergence
    def get_histogram(imgs):
        return np.histogram(imgs.flatten(), bins=256, range=(0, 1), density=True)[0] + alpha

    p_real = get_histogram(real_images)
    p_fake = get_histogram(generated_images)

    js = js_divergence(p_real, p_fake)   # now uses Section 2B definition
    kl = kl_divergence(p_real, p_fake)   # now uses Section 2B definition

    # Metric 3: Diversity Score
    diversity = compute_sample_diversity(generated_images)

    results = {
        "FID": fid,
        "JS_Divergence": js,
        "KL_Divergence": kl,
        "Diversity_Score": diversity
    }

    log_message(f"[Evaluation] FID: {fid:.4f} | JS: {js:.4f} | KL: {kl:.4f} | Diversity: {diversity:.6f}")

    return results



# =====================================
# SECTION 7: Classifier Utility Evaluation (PyTorch)
# =====================================

class SimpleCNNClassifier(nn.Module):
    """
    Simple CNN Classifier for evaluating the utility of synthetic data.
    """
    def __init__(self, input_channels, num_classes):
        super(SimpleCNNClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # pool only over height
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1))   # pool only over height
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (IMG_SHAPE[1] // 4) * IMG_SHAPE[2], 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def train_classifier(model, train_loader, val_loader, device, epochs=20, lr=1e-3, log_interval=5, writer=None):

    """
    Train the CNN classifier with early stopping on validation loss.

    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        device: 'cuda' or 'cpu'
        epochs: Maximum epochs
        lr: Learning rate
        log_interval: Logging frequency

    Returns:
        model: Trained model
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(val_loader.dataset)

        log_message(f"[Classifier Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if writer is not None:
            writer.add_scalar("Classifier/Train_Loss", train_loss, epoch)
            writer.add_scalar("Classifier/Val_Loss", val_loss, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_classifier.pt"))
            log_message("Saved best classifier model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log_message("Early stopping triggered for classifier.")
                break

    # Load best weights before returning
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_classifier.pt")))
    return model

def evaluate_classifier(model, test_loader, device):
    """
    Evaluate a trained classifier with detailed metrics.

    Args:
        model: Trained PyTorch model
        test_loader: Test DataLoader
        device: 'cuda' or 'cpu'

    Returns:
        dict: accuracy, precision, recall, f1, confusion_matrix
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = y_batch.numpy()
            all_preds.append(preds)
            all_labels.append(labels)

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "f1": f1_score(y_true, y_pred, average='macro', zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }
    log_message(f"[Classifier Evaluation] Acc: {metrics['accuracy']:.4f} | Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")

    return metrics



# =====================================
# SECTION 8: Main Runner Script
# =====================================

if __name__ == "__main__":

    # -----------------------------
    # Select Device
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"Using device: {device}")

    # -----------------------------
    # Set Random Seeds for Reproducibility
    # -----------------------------
    set_random_seeds()

    # -----------------------------
    # Load and Prepare Dataset
    # -----------------------------
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_malware_dataset(
        data_path=DATA_PATH,
        img_shape=IMG_SHAPE,
        num_classes=NUM_CLASSES,
        val_fraction=0.5
    )

    # Reshape flattened data back to [N, C, H, W] for CNN classifier
    x_train_cnn = x_train.reshape(-1, IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2])
    x_val_cnn = x_val.reshape(-1, IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2])
    x_test_cnn = x_test.reshape(-1, IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2])

    train_loader, val_loader, test_loader = get_data_loaders(
        x_train, y_train,
        x_val, y_val,
        x_test, y_test,
        batch_size=BATCH_SIZE
    )

    # -----------------------------
    # Build and Train MAF Model
    # -----------------------------
    maf_model = build_maf_model(
        input_dim=np.prod(IMG_SHAPE),
        num_layers=NUM_FLOW_LAYERS
    ).to(device)

    writer = SummaryWriter(log_dir=LOG_DIR)

    maf_model = train_maf_model(
        model=maf_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        writer=writer,
        checkpoint_dir=CHECKPOINT_DIR
    )

    # -----------------------------
    # Generate and Save Synthetic Samples
    # -----------------------------
    synthetic_samples = sample_from_maf(
        model=maf_model,
        num_samples=1000,
        input_dim=np.prod(IMG_SHAPE),
        device=device
    )

    save_synthetic_samples(
        synthetic_samples,
        save_dir=SYNTHETIC_DIR,
        prefix="maf"
    )

    visualize_samples(
        synthetic_samples,
        img_shape=IMG_SHAPE,
        num_display=16,
        save_path=os.path.join(BASE_DIR, "maf_generated_samples.png")
    )

    # -----------------------------
    # Evaluate Generated Samples
    # -----------------------------
    eval_metrics = evaluate_generation_quality(
        real_images=x_val[:1000].reshape(-1, *IMG_SHAPE),
        generated_images=synthetic_samples.reshape(-1, *IMG_SHAPE)
    )
    log_message(f"MAF Evaluation Metrics: {eval_metrics}")

    # -----------------------------
    # Classifier Utility Evaluation
    # -----------------------------
    # (1) Train classifier on real data only
    clf_real = SimpleCNNClassifier(input_channels=IMG_SHAPE[0], num_classes=NUM_CLASSES).to(device)

    # Create classifier dataloaders with reshaped data
    train_loader_cnn, val_loader_cnn, test_loader_cnn = get_data_loaders(
        x_train_cnn, y_train,
        x_val_cnn, y_val,
        x_test_cnn, y_test,
        batch_size=BATCH_SIZE
    )

    clf_real = train_classifier(
        model=clf_real,
        train_loader=train_loader_cnn,
        val_loader=val_loader_cnn,
        device=device,
        epochs=20,
        lr=LR,
        writer=writer
    )

    real_metrics = evaluate_classifier(
        model=clf_real,
        test_loader=test_loader_cnn,  # FIXED
        device=device
    )

    log_message(f"[Classifier Utility - Real Only] {real_metrics}")

    # (2) Train classifier on real + synthetic data
    y_synth = y_train[:synthetic_samples.shape[0]]  # reuse existing labels for simplicity

    x_combined = np.concatenate([x_train, synthetic_samples], axis=0)

    # Ensure labels are converted to NumPy before concatenation
    y_combined = np.concatenate([y_train.numpy(), y_synth.numpy()], axis=0)
    y_combined = torch.from_numpy(y_combined).long() # Convert y_combined to torch tensor

    # FIX: Reshape for CNN
    x_combined_cnn = x_combined.reshape(-1, IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2])
    x_combined_cnn = torch.from_numpy(x_combined_cnn).float()  # convert to tensor

    combined_loader, _, _ = get_data_loaders(
        x_combined_cnn, y_combined,
        x_val_cnn, y_val,
        x_test_cnn, y_test,
        batch_size=BATCH_SIZE
    )

    clf_combined = SimpleCNNClassifier(input_channels=IMG_SHAPE[0], num_classes=NUM_CLASSES).to(device)

    clf_combined = train_classifier(
        model=clf_combined,
        train_loader=combined_loader,
        val_loader=val_loader_cnn,
        device=device,
        epochs=20,
        lr=LR,
        writer=writer
    )

    combined_metrics = evaluate_classifier(
        model=clf_combined,
        test_loader=test_loader_cnn,  # FIXED
        device=device
    )

    log_message(f"[Classifier Utility - Real + Synthetic] {combined_metrics}")

    # -----------------------------
    # Save Final Comparison Results
    # -----------------------------
    comparison = {
        "accuracy_real_only": real_metrics["accuracy"],
        "accuracy_real_plus_synthetic": combined_metrics["accuracy"]
    }

    with open(os.path.join(BASE_DIR, "maf_classifier_comparison.txt"), "w") as f:
        for key, value in comparison.items():
            f.write(f"{key}: {value}\n")

    log_message("Finished MAF PyTorch pipeline execution.")
