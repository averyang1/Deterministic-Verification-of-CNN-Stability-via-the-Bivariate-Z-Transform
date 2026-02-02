import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models

def analyze_kernel_spectral_signature(kernel_matrix, resolution=100):
    """
    Computes the 2D Frequency Response and identifies Zero-Varieties
    of a convolutional kernel using the Bivariate Z-Transform.
    """
    # 1. Establish the 2D Frequency Grid (Sampling the Unit Bi-Circle)
    # This maps the complex variables z1, z2 where |z1|=1 and |z2|=1
    w1 = np.linspace(-np.pi, np.pi, resolution)
    w2 = np.linspace(-np.pi, np.pi, resolution)
    W1, W2 = np.meshgrid(w1, w2)

    # 2. Compute the Bivariate Z-Transform (evaluated on the unit bi-circle)
    # H(e^jw1, e^jw2) = sum sum h[n1, n2] * exp(-j(w1n1 + w2n2))
    h = kernel_matrix
    H = np.zeros((resolution, resolution), dtype=complex)
    
    # Iterate through the discrete spatial support (e.g., 3x3 or 5x5)
    for n1 in range(h.shape[0]):
        for n2 in range(h.shape[1]):
            # The z^-1 operator represents a spatial shift/unit delay
            H += h[n1, n2] * np.exp(-1j * (W1 * n1 + W2 * n2))

    # 3. Calculate Magnitude Response and Zero-Variety
    magnitude_response = np.abs(H)
    
    # Zero-Variety Identification: areas where the response is near zero
    # Geometrically, these define the 'stopbands' or frequencies the kernel ignores
    epsilon = magnitude_response.min() * 1.5  # Threshold for 'zero' regions
    zero_variety_mask = magnitude_response < epsilon

    return magnitude_response, zero_variety_mask

# --- Execution Workflow ---

# A. Extract a learned kernel from a pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)
# Isolating a specific kernel from the first convolutional layer
# Layer 1 typically learns fundamental features like edges
learned_kernel = model.conv1.weight[0, 0].detach().numpy() 

# B. Process through the Spectral Pipeline
mag, zeros = analyze_kernel_spectral_signature(learned_kernel)

# C. Visualization of the "Spectral DNA"
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Magnitude Response (The Passbands/Stopbands)
im1 = ax[0].imshow(mag, extent=[-np.pi, np.pi, -np.pi, np.pi])
ax[0].set_title("2D Magnitude Response $|H(\omega_1, \omega_2)|$")
plt.colorbar(im1, ax=ax[0])

# Plot 2: Zero-Variety Map
# This highlights the geometric distribution of zeros relative to the unit bi-circle
ax[1].imshow(zeros, extent=[-np.pi, np.pi, -np.pi, np.pi], cmap='binary')
ax[1].set_title("Geometric Signature of Zero-Variety")

plt.show()



