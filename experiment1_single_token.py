import os
import math
import torch
import numpy as np
from safetensors import safe_open

MODEL = "granite-4.0-h-350m/model.safetensors"
TOKEN_ID = 52341  # Change this to try different tokens.

# -------------------------------------------------------
# Load one embedding vector only.
# -------------------------------------------------------

with safe_open(MODEL, framework="pt", device="cpu") as f:
    vec = f.get_tensor("model.embed_tokens.weight")[TOKEN_ID].float()

x = np.arange(768)
y = vec.numpy()

print(f"Loaded token {TOKEN_ID}")

# -------------------------------------------------------
# Polynomial fit
# -------------------------------------------------------

def poly_fit(degree):
    coeffs = np.polyfit(x, y, degree)
    pred = np.polyval(coeffs, x)
    rmse = np.sqrt(np.mean((pred - y) ** 2))
    bytes_used = len(coeffs) * 2
    return rmse, bytes_used

# -------------------------------------------------------
# Fourier fit
# -------------------------------------------------------

def fourier_fit(k):
    fft = np.fft.rfft(y)

    keep = np.zeros_like(fft)
    keep[:k] = fft[:k]

    pred = np.fft.irfft(keep, n=len(y))
    rmse = np.sqrt(np.mean((pred - y) ** 2))

    bytes_used = k * 2 * 2   # real + imag BF16

    return rmse, bytes_used

# -------------------------------------------------------
# Cubic spline
# -------------------------------------------------------

from scipy.interpolate import CubicSpline

def spline_fit(control_points):

    idx = np.linspace(0, 767, control_points).astype(int)

    spline = CubicSpline(idx, y[idx])

    pred = spline(x)

    rmse = np.sqrt(np.mean((pred - y) ** 2))

    bytes_used = control_points * 2

    return rmse, bytes_used

print("\nPolynomial")
for d in [4,8,16,32]:
    rmse,b = poly_fit(d)
    print(f"degree={d:2d} bytes={b:3d} RMSE={rmse:.6f}")

print("\nFourier")
for k in [8,16,32,64,128]:
    rmse,b = fourier_fit(k)
    print(f"coeffs={k:3d} bytes={b:3d} RMSE={rmse:.6f}")

print("\nSpline")
for c in [16,32,48,64,96]:
    rmse,b = spline_fit(c)
    print(f"points={c:3d} bytes={b:3d} RMSE={rmse:.6f}")
