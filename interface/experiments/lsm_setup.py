import numpy as np


def spectral_radius(A):
    eigenvalues = np.linalg.eigvals(A)
    return np.max(np.abs(eigenvalues))
    
def generate_liquid_weights(size, minimum=0, maximum=1, connectivity=0.25, scalar=0.5):
    w = np.zeros((size, size))
    
    connections = np.random.rand(size, size) < connectivity
    
    weights = np.abs(np.random.normal(minimum, maximum, (size, size)))
    w[connections] = weights[connections]
    
    np.fill_diagonal(w, 0)
    
    # spectral radius near 1
    w /= spectral_radius(w) * scalar

    return w
