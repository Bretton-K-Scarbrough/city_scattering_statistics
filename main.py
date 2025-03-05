from propagators import propTF_Fresnel
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Physical Constants
lam = 532e-9
L = 500e-3
N = 1000
dx = L / N
x = np.linspace(-L / 2, L / 2 - dx, N)
X, Y = np.meshgrid(x, x)

# Gaussian parameters
# NOTE: Don't change sigma to be larger than ~ 2, increases PSD and causes FFT aliasing
mu, sigma = 0, 1.9


# Create a planewave
planewave = np.exp(-np.pi * (X**2 + Y**2) / 100e-3)

# Generate a random phase distribution
rand_phi = np.random.normal(mu, sigma, planewave.shape)

scattered_field = planewave * np.exp(1j * rand_phi)

propagated_field = propTF_Fresnel(scattered_field, L, lam, 20 * lam)

loops = 100
fig = plt.figure(figsize=(15, 5))
ax3d = fig.add_subplot(131, projection="3d")
for i in range(loops):
    rand_phi = np.random.normal(mu, sigma, planewave.shape)
    scattered_field = propagated_field * np.exp(1j * rand_phi)
    propagated_field = propTF_Fresnel(scattered_field, L, lam, 1000 * lam)
    ax3d.clear()

    im_data = np.imag(propagated_field).flatten()
    re_data = np.real(propagated_field).flatten()
    hist, xedges, yedges = np.histogram2d(re_data, im_data, bins=80)

    # Convert bin edges to centers for 3D plotting
    x_centers = (xedges[:-1] + xedges[1:]) / 2
    y_centers = (yedges[:-1] + yedges[1:]) / 2
    x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing="ij")

    x_pos = x_grid.ravel()
    y_pos = y_grid.ravel()
    z_pos = np.zeros_like(x_pos)
    dz = hist.ravel()

    dx = dy = (xedges[1] - xedges[0]) * np.ones_like(dz)

    ax3d.bar3d(x_pos, y_pos, z_pos, dx, dy, dz, alpha=0.5, shade=True)
    ax3d.set_xlabel("Re")
    ax3d.set_ylabel("Im")

    ax_xslice = fig.add_subplot(132)
    hist_x = np.sum(hist, axis=1)
    ax_xslice.bar(
        x_centers, hist_x, width=(xedges[1] - xedges[0]), color="r", alpha=0.7
    )
    ax_xslice.set_xlabel("Real Part")
    ax_xslice.set_ylabel("Frequency")
    ax_xslice.set_title("X-Slice (Real Part Histogram)")

    ax_yslice = fig.add_subplot(133)
    hist_y = np.sum(hist, axis=0)
    ax_yslice.bar(
        y_centers, hist_y, width=(yedges[1] - yedges[0]), color="g", alpha=0.7
    )
    ax_yslice.set_xlabel("Imaginary Part")
    ax_yslice.set_ylabel("Frequency")
    ax_yslice.set_title("Y-Slice (Imaginary Part Histogram)")

    # plt.title(f"loop {i}")
    plt.pause(1)
plt.show()

# TODO:
# - Create a plane wave
# - Sample a random process for a "phase mask"
#   - Look into the randomness of 3D printing models
# - Simulate the propagation of the plane wave modulated by the phase mask
# - Recursively call the propagation of the wave through a new phase mask every distance d
