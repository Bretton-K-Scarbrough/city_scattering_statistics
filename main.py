from propagators import propTF_Fresnel
import numpy as np
import matplotlib.pyplot as plt
import os

# Physical Constants
lam = 532e-9
L = 500e-3
N = 2000
dx = L / N
x = np.linspace(-L / 2, L / 2 - dx, N)
X, Y = np.meshgrid(x, x)

# Initialize RNG
mu, sigma = 0, 1.9


if __name__ == "__main__":
    simulation_name = "3_4_25_testing"
    file_path = "./figures/" + simulation_name + "/"

    # Generate folder to save things in
    try:
        os.mkdir(file_path)
        print(f"Created '{file_path}' to save data.")
    except FileExistsError:
        print(f"Data will be saved in '{file_path}'")
    except PermissionError:
        print(f"Permission denied: Unable to create '{file_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Create a planewave
    planewave = np.exp(-np.pi * (X**2 + Y**2) / 100e-3)

    # Generate a random phase distribution
    rand_phi = np.random.normal(mu, sigma, planewave.shape)

    scattered_field = planewave * np.exp(1j * rand_phi)

    propagated_field = propTF_Fresnel(scattered_field, L, lam, 20 * lam)

    loops = 100
    plt.figure()
    for i in range(loops):
        rand_phi = np.random.normal(mu, sigma, planewave.shape)
        scattered_field = propagated_field * np.exp(1j * rand_phi)
        propagated_field = propTF_Fresnel(scattered_field, L, lam, 100 * lam)
        prop_intensity = np.abs(propagated_field) ** 2
        plt.clf()
        # plt.hist(prop_intensity.flatten(), bins=100, alpha=0.3, label=f"loop {i}")
        # plt.legend()
        plt.imshow(np.angle(propagated_field), cmap="rainbow")
        plt.title(f"loop {i}")
        plt.pause(1)
    plt.show()

# TODO:
# - Create a plane wave
# - Sample a random process for a "phase mask"
#   - Look into the randomness of 3D printing models
# - Simulate the propagation of the plane wave modulated by the phase mask
# - Recursively call the propagation of the wave through a new phase mask every distance d
