import numpy as np
from typing import Tuple


def propTF_Fresnel(u1: np.ndarray, L: float, lam: float, z: float) -> np.ndarray:
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


def propFraunhofer(
    u1: np.ndarray, L1: float, lam: float, z: float
) -> Tuple[np.ndarray, float]:
    """
    Propagates some field, u1, to z using the Fraunhofer kernel. Note: this is the same as propagating to far field or through a lens. Assumes a monochromatic field distribution

    Parameters:
    u1        : np.ndarray (M,M), complex|real
                2D complex|real array Input field distribution
    L1        : Physical length of u1
    lam       : wavelength of field
    z         : propagation distance (or focal length)

    Returns:
    Tuple[np.ndarray, float]
    Returns:
        u2 : np.ndarray (M,M), complex
             2D complex array Propagated field in the observation plane.
        L2 : float
             Physical length in the observation plane.
    """
    assert u1.ndim == 2, "Input field u1 must be a 2D array"
    assert L1 > 0, "Physical length L must be positive"
    assert lam > 0, "Wavelength lam must be positive"
    assert z != 0, "Propagation distance z cannot be zero"

    M = np.shape(u1)[0]
    dx1 = L1 / M
    k = 2 * np.pi / lam

    L2 = lam * z / dx1  # side length in obs plane
    x2 = np.linspace(-L2 / 2, L2 / 2, M)

    X2, Y2 = np.meshgrid(x2, x2)

    c = 1 / (1j * lam * z) * np.exp(1j * k / (2 * z) * (X2**2 + Y2**2))
    u2 = c * np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(u1))) * dx1**2

    return u2, L2
