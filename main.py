from propagators import propFraunhofer, propTF_Fresnel
import numpy as np
import matplotlib.pyplot as plt
import os

# Physical Constants
lam = 532e-9
L = 500e-3
N = 1000
dx = L / N
x = np.linspace(-L / 2, L / 2 - dx, N)
X, Y = np.meshgrid(x, x)

# Initialize RNG
mu, sigma = 0, 10


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
    A = np.ones_like(X, dtype="complex")
    planewave = A * np.exp(1j * (2 * np.pi) / lam)

    # Generate a random phase distribution
    rand_phi = np.random.normal(mu, sigma, planewave.shape)

    scattered_field = planewave * np.exp(1j * rand_phi)

    propagated_field = propTF_Fresnel(scattered_field, L, lam, 20 * lam)

    plt.figure()
    plt.imshow(np.abs(propagated_field) ** 2)
    plt.show()

    loops = 3
    for i in range(loops):
        rand_phi = np.random.normal(mu, sigma, planewave.shape)
        scattered_filed = propagated_field * np.exp(rand_phi)
        propagated_field = propTF_Fresnel(scattered_field, L, lam, 20 * lam)
        plt.imshow(np.abs(propagated_field) ** 2)
        plt.title(f"Loop {i}")
        plt.show()


# TODO:
# - Create a plane wave
# - Sample a random process for a "phase mask"
#   - Look into the randomness of 3D printing models
# - Simulate the propagation of the plane wave modulated by the phase mask
# - Recursively call the propagation of the wave through a new phase mask every distance d
