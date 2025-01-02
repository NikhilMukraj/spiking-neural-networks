import numpy as np


# spectral radius of liquid should be ~1
def spectral_radius(A):
    eigenvalues = np.linalg.eigvals(A)
    return np.max(np.abs(eigenvalues))

# def generate_liquid_weights(rows, cols):
#   ...
