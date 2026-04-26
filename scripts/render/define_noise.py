# Define noise from experimental images

import numpy as np
from pathlib import Path
import tifffile  # or PIL, cv2, whatever your images are in
import matplotlib.pyplot as plt


# Load stack of N static frames, same camera, no scene change
frames = np.stack([tifffile.imread(f) for f in sorted(Path('...').glob('*.tif'))])
# shape: (N, 4608, 5328)

# Estimate true scene (mean over time)
mean_scene = frames.mean(axis=0)

# Noise realisations
residuals = frames - mean_scene  # shape (N, 4608, 5328)

# Per-pixel temporal noise standard deviation
sigma_map = residuals.std(axis=0)  # shape (4608, 5328)

# Global noise floor summary
print(f"Mean sigma: {sigma_map.mean():.2f} ADU")
print(f"Std of sigma map: {sigma_map.std():.2f} ADU")  # spatial variation

# 1. Global noise level (single number)
sigma_global = residuals.std()

# 2. Spatial noise map — is it uniform or structured?
sigma_map = residuals.std(axis=0)

# 3. Noise distribution — is it Gaussian?
plt.hist(residuals.flatten(), bins=200, density=True)
# Should look Gaussian if dominated by read + shot noise

# 4. Power spectral density — any spatial correlation?
noise_frame = residuals[0]  # single noise realisation
psd = np.abs(np.fft.fftshift(np.fft.fft2(noise_frame)))**2
# Plot log(psd) — flat = white noise, structure = correlated noise

# 5. Row/column profiles of sigma_map — reveals FPN stripes
plt.plot(sigma_map.mean(axis=0))  # column-wise mean → column FPN
plt.plot(sigma_map.mean(axis=1))  # row-wise mean    → row FPN