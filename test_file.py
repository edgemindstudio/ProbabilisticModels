import torch

print("PyTorch Version:", torch.__version__)
print("MPS Available:", torch.backends.mps.is_available())
print("Using MPS:", torch.device("mps"))

