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

def generate_start_firing(cue_firing_rate):
    def start_firing(neuron):
        neuron.chance_of_firing = cue_firing_rate

        return neuron

    return start_firing

def stop_firing(neuron):
    neuron.chance_of_firing = 0

    return neuron


def determine_return_to_baseline(voltages, settling_period, on_phase, off_phase, tolerance):
    baseline = np.array(voltages[1000:off_phase]).mean()

    for i in range(off_phase):
        current_voltage_average = np.array(voltages[off_phase + on_phase + i:]).mean()
        if abs(baseline - current_voltage_average) < tolerance:
            return i
    
    return off_phase
