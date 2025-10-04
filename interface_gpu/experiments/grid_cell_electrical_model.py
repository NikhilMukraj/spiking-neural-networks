# wrap weights funtion around axis in a circular fashion
# https://www.mathworks.com/matlabcentral/answers/1716995-revolving-a-curve-about-the-y-axis-to-generate-a-3d-surface

import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import lixirnet as ln
import argparse
import json


# number of neurons
n = 60

def torodial_dist(a, b, n):
    x1, y1 = a
    x2, y2 = b
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    if dx > n / 2:
        dx = n - dx

    if dy > n / 2:
        dy = n - dy

    return np.sqrt(dx**2 + dy**2)

sigmoid_second_derivative = lambda x: -1 * ((np.exp(x) * (np.exp(x) - 1)) / (np.exp(x) + 1) ** 3)

# sigmoid_second_derivative(torodial_dist(x, y, n))

def setup_neuron(neuron):
    neuron.current_voltage = np.random.uniform(neuron.c, neuron.v_th)
    neuron.c_m = 25

    return neuron

# direction 0 is right, direction 1 is left
def setup_poisson_given_direction(direction):
    def setup_poisson(pos, neuron):
        if pos[0] == direction:
            neuron.rate = 0.01
        else:
            neuron.rate = 0
        
        return neuron

    return setup_poisson

def find_peaks_above_threshold(series, threshold):
    peaks, _ = scipy.signal.find_peaks(np.array(series))
    filtered_peaks = [index for index in peaks if series[index] > threshold]
    
    return filtered_peaks

# weights for head direction layer
grid_weight = lambda x, y: 3 * (np.exp(-2 * torodial_distance(x, y, n) ** 2 / (n * 3))) - 0.9
# head direction feedback into shift layers to engage in shifting behavior when turning cells are on
grid_to_shift_weight = lambda x, y: 1 * (np.exp(-2 * torodial_distance(x, y, n) ** 2 / (n * 3)) - 0.2)

grid_cells = ln.IzhikevichNeuronLattice(grid_cells_ring)
grid_cells.populate(exc_neuron, n, n)
grid_cells.connect(lambda x, y: True, grid_weight)
grid_cells.apply(setup_neuron)
grid_cells.update_grid_history = True
