import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import lixirnet as ln


sns.set_theme(style='darkgrid')

n = 7
num = n * n

def get_weights(n, patterns, a=0, b=0, scalar=1):
    w = np.zeros([n, n])
    for pattern in patterns:
        for i in range(n):
            for j in range(n):
                w[i][j] += (pattern[i] - b) * (pattern[j] - a) 
    for diag in range(n):
        w[diag][diag] = 0

    w *= scalar
    
    return w

def check_uniqueness(patterns):
    for n1, i in enumerate(patterns):
        for n2, j in enumerate(patterns):
            if n1 != n2 and (np.array_equal(i, j) or np.array_equal(np.logical_not(i).astype(int), j)):
                return True
    
    return False

def calculate_correlation(patterns):
    num_patterns = patterns.shape[0]
    correlation_matrix = np.zeros((num_patterns, num_patterns))
    
    for i in range(num_patterns):
        for j in range(i, num_patterns):
            correlation = np.dot(patterns[i], patterns[j])
            correlation_matrix[i, j] = correlation
            correlation_matrix[j, i] = correlation
    
    return correlation_matrix

p_on = 0.5
num_patterns = 4

not_unique = True
too_correlated = True
while not_unique or too_correlated:
    patterns = []
    for i in range(num_patterns):
        p = np.random.binomial(1, p_on, n * n)
        # p = p * 2 - 1

        patterns.append(p)

    not_unique = check_uniqueness(patterns)    
    too_correlated = calculate_correlation(np.array(patterns)).sum() > 150

# for i in patterns:
#     plt.imshow(i.reshape(n, n))
#     plt.axis('off')
#     plt.colorbar()
#     plt.show()

w = get_weights(num, patterns, 1, 1, 0.5 / num_patterns)

# plt.imshow(w)
# plt.axis('off')
# plt.colorbar()
# plt.show()

def setup_neuron(neuron):
    neuron.current_voltage = np.random.uniform(-65, 30)
    neuron.gap_conductance = 10
    return neuron

def get_spike_train_setup_function(pattern_index, distortion):
    def setup_spike_train(pos, neuron):
        x, y = pos
        index = x * n + y
        state = patterns[pattern_index][index] == 1

        if np.random.uniform(0, 1) < distortion:
            state ^= 1

        if state:
            neuron.chance_of_firing = 0.01
        else:
            neuron.chance_of_firing = 0

        return neuron

    return setup_spike_train

def find_peaks_above_threshold(series, threshold):
    peaks, _ = scipy.signal.find_peaks(np.array(series))
    filtered_peaks = [index for index in peaks if series[index] > threshold]
    
    return filtered_peaks

inh_lattice = ln.IzhikevichLattice(0)
inh_lattice.populate(ln.IzhikevichNeuron(), 3, 3)
inh_lattice.apply(setup_neuron)
inh_lattice.connect(lambda x, y: x != y, lambda x, y: -1)

exc_lattice = ln.IzhikevichLattice(1)
exc_lattice.populate(ln.IzhikevichNeuron(), n, n)
exc_lattice.apply(setup_neuron)
position_to_index = exc_lattice.position_to_index
exc_lattice.connect(
    lambda x, y: bool(w[position_to_index[x]][position_to_index[y]] != 0), 
    lambda x, y: w[position_to_index[x]][position_to_index[y]],
)
exc_lattice.update_grid_history = True

spike_train_lattice = ln.PoissonLattice(2)
spike_train_lattice.populate(ln.PoissonNeuron(), n, n)
spike_train_lattice.apply_given_position(get_spike_train_setup_function(0, 0.1))

network = ln.IzhikevichNetwork.generate_network([exc_lattice, inh_lattice], [spike_train_lattice])
network.connect(0, 1, lambda x, y: True, lambda x, y: -2)
network.connect(1, 0, lambda x, y: True, lambda x, y: 3)
network.connect(2, 1, lambda x, y: x == y, lambda x, y: 5)

network.set_dt(0.5)

for _ in tqdm(range(1_000)):
    network.run_lattices(1)

hist = network.get_lattice(1).history
data = [i.flatten() for i in np.array(hist)]
peaks = [find_peaks_above_threshold([j[i] for j in data], 20) for i in range(len(data[0]))]

def acc(true_pattern, pred_pattern, threshold=10): 
    current_pred_pattern = pred_pattern
    current_pred_pattern[pred_pattern < threshold] = 0 
    current_pred_pattern[pred_pattern >= threshold] = 1
    return (true_pattern.reshape(n, n) == current_pred_pattern.reshape(n, n)).sum() / (n * n)

current_pred_pattern = np.array([len(i) for i in peaks])
firing_max = current_pred_pattern.max()
step = 1
accuracy = max(
    [acc(patterns[0], np.array([len(i) for i in peaks]), threshold=i) for i in range(0, firing_max, step)]
)

plt.imshow(np.array([len(i) for i in peaks]).reshape(n, n))
plt.axis('off')
plt.colorbar()
plt.title(f'Firing Rate Heatmap, Accuracy: {accuracy * 100:.2f}')
plt.show()
