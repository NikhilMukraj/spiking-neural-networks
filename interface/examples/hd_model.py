import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import lixirnet as ln


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
    neuron.current_voltage = np.random.uniform(neuron.v_init, neuron.v_th)

    return neuron

# direction 0 is right, direction 1 is left
def setup_poisson_given_direction(direction):
    def setup_poisson(pos, neuron):
        if pos[0] == direction:
            neuron.chance_of_firing = 0.005
        else:
            neuron.chance_of_firing = 0
        
        return neuron

    return setup_poisson

def find_peaks_above_threshold(series, threshold):
    peaks, _ = scipy.signal.find_peaks(np.array(series))
    filtered_peaks = [index for index in peaks if series[index] > threshold]
    
    return filtered_peaks

# weights for head direction layer
hd_weight = lambda x, y: 3 * (np.exp(-2 * ring_distance(n, x[0], y[0]) ** 2 / (n * 10))) - 0.9
# head direction feedback into shift layers to engage in shifting behavior when turning cells are on
hd_to_shift_weight = lambda x, y: 1 * (np.exp(-2 * ring_distance(n, x[0], y[0]) ** 2 / (n * 10)) - 0.2)

sigmoid_second_derivative = lambda x: -1 * ((np.exp(x) * (np.exp(x) - 1)) / (np.exp(x) + 1) ** 3)

# inhibits neurons in opposite direction, activates in desired direction
shift_left_weight = lambda x, y: 20 * sigmoid_second_derivative(signed_ring_distance(x, y) / 5)
shift_right_weight = lambda x, y: -20 * sigmoid_second_derivative(signed_ring_distance(x, y) / 5)

shift_left = ln.IzhikevichLattice(0)
shift_left.populate(ln.IzhikevichNeuron(), n, 1)
shift_left.apply(setup_neuron)
shift_left.update_grid_history = True

turning_cells = ln.PoissonLattice(3)
turning_cells.populate(ln.PoissonNeuron(), 2, 1)
turning_cells.apply_given_position(setup_poisson_given_direction(0))
turning_cells.update_grid_history = True

shift_right = ln.IzhikevichLattice(1)
shift_right.populate(ln.IzhikevichNeuron(), n, 1)
shift_right.apply(setup_neuron)
shift_right.update_grid_history = True

hd = ln.IzhikevichLattice(2)
hd.populate(ln.IzhikevichNeuron(), n, 1)
hd.connect(lambda x, y: True, hd_weight)
hd.apply(setup_neuron)
hd.update_grid_history = True

head_direction_attractor = ln.IzhikevichNetwork.generate_network([shift_left, shift_right, hd], [turning_cells])
head_direction_attractor.connect(3, 0, lambda x, y: True, lambda x, y: 10)
head_direction_attractor.connect(3, 1, lambda x, y: True, lambda x, y: 10)
head_direction_attractor.connect(0, 2, lambda x, y: True, shift_right_weight)
head_direction_attractor.connect(1, 2, lambda x, y: True, shift_left_weight)
head_direction_attractor.connect(2, 0, lambda x, y: True, hd_to_shift_weight)
head_direction_attractor.connect(2, 1, lambda x, y: True, hd_to_shift_weight)
head_direction_attractor.set_dt(1)
head_direction_attractor.parallel = True

for _ in tqdm(range(1_000)):
    head_direction_attractor.run_lattices(1)

peak_threshold = 20

hist = head_direction_attractor.get_lattice(2).history
data = [i.flatten() for i in np.array(hist)]
peaks = [find_peaks_above_threshold([j[i] for j in data], peak_threshold) for i in range(len(data[0]))]

window = 100
firing_rates = {}
for i in range(0, 1_000, window):
    firing_rates[i] = {}
    for n, peak_array in enumerate(peaks):
        firing_rates[i][n] = len([j for j in peak_array if j <= i and j > i - window])

thetas = []

for key, value in firing_rates.items():
    arg_max = 0
    max_value = 0
    for inner_key, inner_value in value.items():
        if inner_value >= max_value:
            max_value = inner_value
            arg_max = inner_key

    thetas.append(arg_max)

# graphs path over time
plt.polar(-np.cumsum(np.array(thetas) * (360 / n)), range(0, 1_000, window))
plt.show()
