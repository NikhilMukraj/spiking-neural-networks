import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import lixirnet as ln
import argparse
import json


parser = argparse.ArgumentParser(description='Electrochemical model of head direction')
parser.add_argument('-i','--iterations', help='Number of iterations', required=False)
parser.add_argument('-f','--file', help='Peaks output file', required=False)
args = parser.parse_args()

sns.set_theme(style='darkgrid')

# number of neurons
n = 60

sigmoid_second_derivative = lambda x: np.exp(x) * (np.exp(x) - 1) / ((np.exp(x) + 1) ** 3)

def circular_displacement(length, theta1, theta2):
    raw_displacement = theta2 - theta1
    normalized_displacement = (raw_displacement + (length / 2)) % length - (length / 2)
    return normalized_displacement

signed_ring_distance = lambda x, y: circular_displacement(n, x[0], y[0])

ring_distance = lambda length, i, j: min(abs(i - j), length - abs(i - j))

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
hd_weight = lambda x, y: 3 * (np.exp(-2 * ring_distance(n, x[0], y[0]) ** 2 / (n * 3))) - 0.9
# head direction feedback into shift layers to engage in shifting behavior when turning cells are on
hd_to_shift_weight = lambda x, y: 1 * (np.exp(-2 * ring_distance(n, x[0], y[0]) ** 2 / (n * 3)) - 0.2)

sigmoid_second_derivative = lambda x: -1 * ((np.exp(x) * (np.exp(x) - 1)) / (np.exp(x) + 1) ** 3)

# inhibits neurons in opposite direction, activates in desired direction
shift_left_weight = lambda x, y: 20 * sigmoid_second_derivative(signed_ring_distance(x, y) / 10)
shift_right_weight = lambda x, y: -20 * sigmoid_second_derivative(signed_ring_distance(x, y) / 10)

glu = ln.GlutamateReceptor()
gabaa = ln.GABAReceptor()
receptors = ln.DopaGluGABA()
receptors.insert(ln.DopaGluGABANeurotransmitterType.Glutamate, glu)
receptors.insert(ln.DopaGluGABANeurotransmitterType.GABA, gabaa)

exc_neuron = ln.IzhikevichNeuron()
exc_neuron.set_synaptic_neurotransmitters({ln.DopaGluGABANeurotransmitterType.Glutamate : ln.BoundedNeurotransmitterKinetics(clearance_constant=0.001)})
exc_neuron.set_receptors(receptors)
inh_neuron = ln.IzhikevichNeuron()
inh_neuron.set_synaptic_neurotransmitters({ln.DopaGluGABANeurotransmitterType.GABA : ln.BoundedNeurotransmitterKinetics(clearance_constant=0.001)})
inh_neuron.set_receptors(receptors)

rate_spike_train = ln.RateSpikeTrain()
rate_spike_train.set_synaptic_neurotransmitters({ln.DopaGluGABANeurotransmitterType.Glutamate : ln.BoundedNeurotransmitterKinetics()})

left_ring = 0
right_ring = 1
hd_ring = 2
turning = 3
left_ring_inh = 4
right_ring_inh = 5
hd_inh_ring = 6

shift_left = ln.IzhikevichNeuronLattice(left_ring)
shift_left.populate(exc_neuron, n, 1)
shift_left.apply(setup_neuron)
shift_left.update_grid_history = True

shift_right = ln.IzhikevichNeuronLattice(right_ring)
shift_right.populate(exc_neuron, n, 1)
shift_right.apply(setup_neuron)
shift_right.update_grid_history = True

shift_left_inh = ln.IzhikevichNeuronLattice(left_ring_inh)
shift_left_inh.populate(inh_neuron, n, 1)
shift_left_inh.apply(setup_neuron)
shift_left_inh.update_grid_history = True

shift_right_inh = ln.IzhikevichNeuronLattice(right_ring_inh)
shift_right_inh.populate(inh_neuron, n, 1)
shift_right_inh.apply(setup_neuron)
shift_right_inh.update_grid_history = True

hd = ln.IzhikevichNeuronLattice(hd_ring)
hd.populate(exc_neuron, n, 1)
hd.connect(lambda x, y: True, hd_weight)
hd.apply(setup_neuron)
hd.update_grid_history = True

hd_inh = ln.IzhikevichNeuronLattice(hd_inh_ring)
hd_inh.populate(inh_neuron, n, 1)
hd_inh.connect(lambda x, y: True, hd_weight)
hd_inh.apply(setup_neuron)
hd_inh.update_grid_history = True

turning_cells = ln.RateSpikeTrainLattice(turning)
turning_cells.populate(rate_spike_train, 2, 1)
turning_cells.apply_given_position(setup_poisson_given_direction(0))
# turning_cells.update_grid_history = True

inh_strength = 2

head_direction_attractor = ln.IzhikevichNeuronNetwork.generate_network([shift_left, shift_right, shift_left_inh, shift_right_inh, hd_inh, hd], [turning_cells])
head_direction_attractor.connect(turning, left_ring, lambda x, y: True, lambda x, y: 10)
# head_direction_attractor.connect(3, 1, lambda x, y: True, lambda x, y: 10)
head_direction_attractor.connect(left_ring, hd_ring, lambda x, y: True, lambda x, y: max(shift_right_weight(x, y), 0))
head_direction_attractor.connect(left_ring, left_ring_inh, lambda x, y: True, lambda x, y: max(-inh_strength * shift_right_weight(x, y), 0))
head_direction_attractor.connect(left_ring_inh, hd_ring, lambda x, y: True, lambda x, y: max(-1 * shift_right_weight(x, y), 0))
head_direction_attractor.connect(right_ring, hd_ring, lambda x, y: True, lambda x, y: max(shift_left_weight(x, y), 0))
head_direction_attractor.connect(right_ring, right_ring_inh, lambda x, y: True, lambda x, y: max(-inh_strength * shift_left_weight(x, y), 0))
head_direction_attractor.connect(right_ring_inh, hd_ring, lambda x, y: True, lambda x, y: max(-1 * shift_left_weight(x, y), 0))
head_direction_attractor.connect(hd_ring, left_ring, lambda x, y: True, lambda x, y: max(hd_to_shift_weight(x, y), 0))
head_direction_attractor.connect(hd_ring, hd_inh_ring, lambda x, y: True, lambda x, y: max(-inh_strength * hd_to_shift_weight(x, y), 0))
head_direction_attractor.connect(hd_inh_ring, left_ring, lambda x, y: True, lambda x, y: max(-1 * hd_to_shift_weight(x, y), 0))
head_direction_attractor.connect(hd_ring, right_ring, lambda x, y: True, lambda x, y: max(hd_to_shift_weight(x, y), 0))
head_direction_attractor.connect(hd_ring, hd_inh_ring, lambda x, y: True, lambda x, y: max(-inh_strength * hd_to_shift_weight(x, y), 0))
head_direction_attractor.connect(hd_inh_ring, right_ring, lambda x, y: True, lambda x, y: max(-1 * hd_to_shift_weight(x, y), 0))
head_direction_attractor.set_dt(1)
head_direction_attractor.electrical_synapse = False
head_direction_attractor.chemical_synapse = True
head_direction_attractor.parallel = True

if args.iterations is None:
    iterations = 10_000
else:
    iterations = int(args.iterations)

for _ in tqdm(range(iterations)):
    head_direction_attractor.run_lattices(1)

# try converting to gpu after generation of network

peak_threshold = 20

hist = head_direction_attractor.get_lattice(hd_ring).history
data = [i.flatten() for i in np.array(hist)]
peaks = [find_peaks_above_threshold([j[i] for j in data], peak_threshold) for i in range(len(data[0]))]

# calculates the center of mass on a ring
def center_of_mass_ring(arr):
    length = len(arr)
    indices = np.arange(length)
    angles = 2 * np.pi * indices / length
    
    x_components = np.sum(np.cos(angles) * arr)
    y_components = np.sum(np.sin(angles) * arr)
    
    angle_of_com = np.arctan2(y_components, x_components)

    if angle_of_com < 0:
        angle_of_com += 2 * np.pi

    center_of_mass_index = (angle_of_com * length) / (2 * np.pi)
    
    return center_of_mass_index

window = 100
firing_rates = {}
for i in range(0, iterations, window):
    firing_rates[i] = {}
    for n, peak_array in enumerate(peaks):
        firing_rates[i][n] = len([j for j in peak_array if j <= i and j > i - window])

thetas = []

for key, value in firing_rates.items():
    thetas.append(center_of_mass_ring(np.array(list(value.values()))))

# graphs the path over time
plt.polar(np.deg2rad(np.array(thetas) * (360 / n)), range(0, iterations, window))
plt.title('Path over Time')
plt.show()

plt.title('Raster Plot')
for peak_index in range(len(peaks)):
    plt.scatter(peaks[peak_index], [peak_index for i in range(len(peaks[peak_index]))], color='black')
plt.show()

if args.file is not None:
    with open(args.file, 'w+') as f:
        json.dump({'peaks' : [[int(item) for item in sublist] for sublist in peaks]}, f)
