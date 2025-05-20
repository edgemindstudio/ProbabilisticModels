# Conditional DCGAN for Malware Traffic Synthesis

This project implements and compares several generative models for malware traffic synthesis using the [USTC-TFC2016 dataset](https://www.unb.ca/cic/datasets/malicious-traffic.html). The models aim to generate realistic synthetic samples that can augment cybersecurity datasets for training robust malware classifiers.

## 🔍 Objective

To understand and compare the performance of different generative models by:

1. Training **DCGAN**, **Conditional DCGAN**, **Wasserstein GAN with Gradient Penalty**, **VAE**, **Autoregressive Models**, and **Diffusion Models** on malware traffic data.
2. Evaluating generative quality using **FID Score**, **discriminator accuracy**, and **visual inspection**.
3. Using generated samples to **train classifiers**, then test on real-world samples to assess generalization power.

## 🧠 Why This Is Important

- GANs can address **class imbalance** in cybersecurity datasets.
- Helps simulate realistic traffic patterns for training IDS/IPS without exposing real malware.
- Provides insights into **how well generative models can learn structured byte-level traffic**.

---

## 📁 Project Structure

```bash

ProbabilisticModels/
│
├── USTC-TFC2016_malware/          # Contains train/test npy files (preprocessed to 40x40x1)
├── dcgan/                         # Deep Convolutional GAN (DCGAN) and WGAN-GP
│   ├── dcgan.py
│   └── wdcgan.py
├── conditional_dcgan/             # Conditional DCGAN (per class generation)
│   └── conditional_dcgan.py
├── vae/                           # Variational Autoencoder (To be implemented)
├── autoregressive/                # Autoregressive Models (To be implemented)
├── diffusion/                     # Diffusion Models (To be implemented)
├── utils/                         # Utility scripts for evaluation, plotting, etc.
├── generated_sample_epoch_*.png  # Sample visual outputs
├── dcgan_loss_plot.png
├── dcgan_fid_plot.png
├── gan_training_log.txt
└── README.md
```

## 📦 Requirements

Make sure to install all dependencies listed in `requirements.txt`:

```bash
  pip install -r requirements.txt