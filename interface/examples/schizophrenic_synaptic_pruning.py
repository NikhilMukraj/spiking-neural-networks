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

# checks if any patterns are equal to another pattern but inverted
def check_uniqueness(patterns):
    for n1, i in enumerate(patterns):
        for n2, j in enumerate(patterns):
            if n1 != n2 and (np.array_equal(i, j) or np.array_equal(np.logical_not(i).astype(int), j)):
                return True
    
    return False

# checks to see if patterns are correlated
# if pattern is too correlated, energies may lead to worse convergence
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

def setup_neuron(neuron):
    neuron.current_voltage = np.random.uniform(-65, 30)
    neuron.gap_conductance = 10
    return neuron

# generates spike train input given a pattern and how much to distort the pattern
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

def acc(true_pattern, pred_pattern, threshold=10): 
    current_pred_pattern = pred_pattern
    current_pred_pattern[pred_pattern < threshold] = 0 
    current_pred_pattern[pred_pattern >= threshold] = 1
    return (true_pattern.reshape(n, n) == current_pred_pattern.reshape(n, n)).sum() / (n * n)

num_iterations = 1_000
trials = 10
peak_threshold = 20

connectivities = [1, 0.8, 0.6, 0.4, 0.2]
distortions = [0.1]

weights = {}
accs = {}

# runs simulation for each connectivity and distortion level for the given amount of trials
# connectivity is varied within the excitatory lattice which represents the stored patterns,
# an accuracy of 1 is given to recalls that remember >=75% of the given pattern, 0 otherwise,
# accuracy is recorded and then graphed 
for connectivity in tqdm(connectivities):
    for _ in tqdm(range(trials)):
        for distortion in distortions:
            for pattern_index, pattern in enumerate(patterns):
                inh_lattice = ln.IzhikevichLattice(0)
                inh_lattice.populate(ln.IzhikevichNeuron(), 3, 3)
                inh_lattice.apply(setup_neuron)
                inh_lattice.connect(lambda x, y: x != y, lambda x, y: -1)

                exc_lattice = ln.IzhikevichLattice(1)
                exc_lattice.populate(ln.IzhikevichNeuron(), n, n)
                exc_lattice.apply(setup_neuron)
                position_to_index = exc_lattice.position_to_index
                exc_lattice.connect(
                    lambda x, y: bool(
                        w[position_to_index[x]][position_to_index[y]] != 0 
                        and np.random.uniform(0, 1) < connectivity
                    ), 
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
                weights[connectivity] = network.get_lattice(1).weights

                for _ in range(num_iterations):
                    network.run_lattices(1)

                hist = network.get_lattice(1).history
                data = [i.flatten() for i in np.array(hist)]
                peaks = [find_peaks_above_threshold([j[i] for j in data], peak_threshold) for i in range(len(data[0]))]

                current_pred_pattern = np.array([len(i) for i in peaks])
                firing_max = current_pred_pattern.max()
                accuracy = max(
                    [acc(patterns[0], np.array([len(i) for i in peaks]), threshold=i) for i in range(0, firing_max)]
                )
                if accuracy < 0.75:
                    accuracy = 0
                else:
                    accuracy = 1

                key = (connectivity, distortion, pattern_index)
                if key not in accs:
                    accs[key] = [accuracy]
                else:
                    accs[key].append(accuracy) 

aggregated_accs = {}

for (connectivity, _, _), accuracies in accs.items():
    if connectivity not in aggregated_accs:
        aggregated_accs[connectivity] = []
    aggregated_accs[connectivity].extend(accuracies)

aggregated_accs = {k: np.array(v).mean() for k, v in aggregated_accs.items()}

# graphs average accuracy for given synaptic connectivity
plt.plot([i * 100 for i in aggregated_accs.keys()], [i * 100 for i in aggregated_accs.values()])
plt.xlabel('Synaptic Connectivity (%)')
plt.ylabel('Accuracy (%)')
plt.title('Synaptic Connectivity versus Accuracy of Memory Recall')
plt.show()
