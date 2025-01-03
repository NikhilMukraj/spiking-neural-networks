import numpy as np


def spectral_radius(A):
    eigenvalues = np.linalg.eigvals(A)
    return np.max(np.abs(eigenvalues))

def generate_liquid_weights(rows, cols, minimum=0, maximum=1):
    x = np.random.uniform(minimum, maximum, (rows, cols))
    np.fill_diagonal(x, 0)

    # spectral radius of liquid should be ~1
    x /= spectral_radius(x)

    return x
