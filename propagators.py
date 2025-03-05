import numpy as np
from typing import Tuple


def propTF_Fresnel(u1: np.ndarray, L: float, lam: float, z: float) -> np.ndarray:
    """
    Computes the Fresnel transfer function propagation of a wavefront.

    This function simulates the propagation of an input field `u1` over a distance `z`
    using the transfer function approach in the Fresnel approximation.

    Parameters:
    -----------
    u1 : np.ndarray
        The input complex field (2D array) representing the initial wavefront.
    L : float
        The side length of the computational domain (physical size of the field in meters).
    lam : float
        The wavelength of the propagating wave (in meters).
    z : float
        The propagation distance (in meters).

    Returns:
    --------
    np.ndarray
        The propagated complex field (2D array) after a distance `z`.
    """
    k = 2 * np.pi / lam
    M = u1.shape[0]
    dx = L / M

    # Create fx coords
    fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx) - (1 / L), M)
    FX, FY = np.meshgrid(fx, fx)

    # Transfer function
    H = np.exp(1j * k * z) * np.exp(-1j * np.pi * lam * z * (FX**2 + FY**2))
    H = np.fft.fftshift(H)

    # FFT and center
    U1 = np.fft.fft2(np.fft.fftshift(u1))

    # Multiply in frequency
    U2 = U1 * H

    # Inverse FFT and center
    u2 = np.fft.ifftshift(np.fft.ifft2(U2))
    return u2
