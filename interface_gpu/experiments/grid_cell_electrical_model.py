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


parser = argparse.ArgumentParser(description='Electrical model of grid cells')
parser.add_argument('-i','--iterations', help='Number of iterations', required=False)
parser.add_argument('-f','--file', help='Peaks output file', required=False)
args = parser.parse_args()

# number of neurons
n = 30

def torodial_dist(a, b, n):
    x1, y1 = a
    x2, y2 = b
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    if dx > n / 2:
        dx = n - dx

    if dy > n / 2:
        dy = n - dy

    return np.sqrt(dx ** 2 + dy ** 2)

sigmoid_second_derivative = lambda x: -1 * ((np.exp(x) * (np.exp(x) - 1)) / (np.exp(x) + 1) ** 3)

def setup_neuron(neuron):
    neuron.current_voltage = np.random.uniform(neuron.c, neuron.v_th)
    neuron.c_m = 25

    return neuron

def setup_poisson_given_coords(x, y):
    def setup_poisson(pos, neuron):
        neuron.rate = 0.01 * (1 / np.clip(torodial_dist(pos, (x, y), n), 1e-10, None))
        
        return neuron

    return setup_poisson

def find_peaks_above_threshold(series, threshold):
    peaks, _ = scipy.signal.find_peaks(np.array(series))
    filtered_peaks = [index for index in peaks if series[index] > threshold]
    
    return filtered_peaks

# weights for grid cell layer
grid_weight = lambda x, y: 3 * (np.exp(-2 * torodial_dist(x, y, n) ** 2 / (n * 3))) - 0.9
# grid cell feedback into shift layers to engage in shifting behavior when turning cells are on
grid_to_shift_weight = lambda x, y: 1 * (np.exp(-2 * torodial_dist(x, y, n) ** 2 / (n * 3)) - 0.2)

exc_neuron = ln.IzhikevichNeuron()

rate_spike_train = ln.RateSpikeTrain()

grid_cells_ring = 0
setters = 1

grid_cells = ln.IzhikevichNeuronLattice(grid_cells_ring)
grid_cells.populate(exc_neuron, n, n)
grid_cells.connect(lambda x, y: True, grid_weight)
grid_cells.apply(setup_neuron)
grid_cells.update_grid_history = True

setting_cells = ln.RateSpikeTrainLattice(setters)
setting_cells.populate(rate_spike_train, n, n)
setting_cells.apply_given_position(setup_poisson_given_coords(0, 0))

grid_attractor = ln.IzhikevichNeuronNetwork.generate_network([grid_cells], [setting_cells])
grid_attractor.connect(setters, grid_cells_ring, lambda x, y: True, lambda x, y: 1)
grid_attractor.parallel = True
grid_attractor.set_dt(1)

for _ in tqdm(range(int(args.iterations) if args.iterations is not None else 1_000)):
    grid_attractor.run_lattices(1)

peak_threshold = 20

hist = head_direction_attractor.get_lattice(hd_ring).history
data = [i.flatten() for i in np.array(hist)]
peaks = [find_peaks_above_threshold([j[i] for j in data], peak_threshold) for i in range(len(data[0]))]
if args.file is not None:
    with open(args.file, 'w+') as f:
        json.dump({'peaks' : [[int(item) for item in sublist] for sublist in peaks]}, f)
