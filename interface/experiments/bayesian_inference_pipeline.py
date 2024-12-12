# bayesian inference using a single on/off cue
# bayesian inference using another memory as a cue
# bayesian inference using d1/d2
# option to enable d1 or d2 on exc or inh group

import toml
import json
import sys
import itertools
import numpy as np
import scipy
from tqdm import tqdm
import lixirnet as ln


def frange(x, y, step):
  while x < y + step:
    yield x
    x += step

def parse_range_or_list(data):
    result = {}

    for key, value in data.items():
        if isinstance(value, dict):
            if 'min' in value and 'max' in value and 'step' in value:
                result[key] = list(frange(value['min'], value['max'], value['step']))
            else:
                result[key] = value
        else:
            result[key] = value

    return result

def parse_toml(f):
    toml_data = toml.load(f)

    parsed_data = {}
    for section, data in toml_data.items():
        parsed_data[section] = parse_range_or_list(data)

    return parsed_data

def fill_defaults(parsed):
    if 'simulation_parameters' not in parsed:
        raise ValueError('Requires `simulation_parameters` table')

    if 'filename' not in parsed['simulation_parameters']:
        raise ValueError('Requires `filename` field in `simulation_parameters`')
    
    if 'variables' not in parsed:
        raise ValueError('Requires `variables` table')

    if 'iterations1' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['iterations1'] = 3_000
    if 'iterations2' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['iterations2'] = 3_000

    if 'peaks_on' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['peaks_on'] = False

    if 'main_firing_rate' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['main_firing_rate'] = 0.01
    if 'bayesian_firing_rate' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['bayesian_firing_rate'] = 0.01
    
    if 'measure_snr' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['measure_snr'] = False
    
    if 'first_window' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['first_window'] = 1_000
    if 'second_window' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['second_window'] = 1_000

    if 'trials' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['trials'] = 10

    if 'num_patterns' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['num_patterns'] = 3
    if 'weights_scalar' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['weights_scalar'] = 1
    if 'inh_weights_scalar' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['inh_weights_scalar'] = 0.25
    if 'a' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['a'] = 1
    if 'b' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['b'] = 1

    if 'correlation_threshold' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['correlation_threshold'] = 0.08

    if 'use_correlation_as_accuracy' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['use_correlation_as_accuracy'] = False
    if 'get_all_accuracies' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['get_all_accuracies'] = False

    if 'skew' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['skew'] = 1

    if 'exc_n' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['exc_n'] = 7
    if 'inh_n' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['inh_n'] = 3

    if 'dt' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['dt'] = 1
    if 'c_m' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['c_m'] = 25

    if 'distortion' not in parsed['variables']:
        parsed['variables']['distortion'] = [0.15]

    if 'prob_of_exc_to_inh' not in parsed['variables']:
        parsed['variables']['prob_of_exc_to_inh'] = [0.5]
    if 'exc_to_inh' not in parsed['variables']:
        parsed['variables']['exc_to_inh'] = [1]
    if 'spike_train_to_exc' not in parsed['variables']:
        parsed['variables']['spike_train_to_exc'] = [5]
    if 'bayesian_to_exc' not in parsed['variables']:
        parsed['variables']['bayesian_to_exc'] = [5]

    if 'nmda_g' not in parsed['variables']:
        parsed['variables']['nmda_g'] = [0.6]
    if 'ampa_g' not in parsed['variables']:
        parsed['variables']['ampa_g'] = [1]
    if 'gabaa_g' not in parsed['variables']:
        parsed['variables']['gabaa_g'] = [1.2]

    if 'glutamate_clearance' not in parsed['variables']:
        parsed['variables']['glutamate_clearance'] = [0.001]
    if 'gabaa_clearance' not in parsed['variables']:
        parsed['variables']['gabaa_clearance'] = [0.001]

    # single on/off cue versus entire group
    # d1/d2

def generate_key_helper(key, parsed, given_key):
    if len(parsed['variables'][given_key]) != 1:
        key.append(f'{given_key}: {current_state[given_key]}')

# *************************
# * THIS CANNOT BE SHARED *
# *************************
def generate_key(parsed, current_state):
    key = []

    key.append(f'trial: {current_state["trial"]}')

    key.append(f'pattern1: {current_state["pattern1"]}')
    key.append(f'pattern2: {current_state["pattern2"]}')

    generate_key_helper(key, parsed, 'distortion')

    generate_key_helper(key, parsed, 'prob_of_exc_to_inh')
    generate_key_helper(key, parsed, 'exc_to_inh')
    generate_key_helper(key, parsed, 'spike_train_to_exc')

    generate_key_helper(key, parsed, 'nmda_g')
    generate_key_helper(key, parsed, 'ampa_g')
    generate_key_helper(key, parsed, 'gabaa_g')

    generate_key_helper(key, parsed, 'glutamate_clearance')
    generate_key_helper(key, parsed, 'gabaa_clearance')

    return ', '.join(key)

def try_max(a):
    if len(a) == 0:
        return 0
    else:
        return max(a)

with open(sys.argv[1], 'r') as f:
    parsed_toml = parse_toml(f)

fill_defaults(parsed_toml)

exc_n = parsed_toml['simulation_parameters']['exc_n']
num = exc_n * exc_n

inh_n = parsed_toml['simulation_parameters']['inh_n']

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

def weights_ie(n, scalar, patterns, num_patterns):
    w = np.zeros([n, n])
    for pattern in patterns:
        for i in range(n):
            for j in range(n):
                w[i][j] += pattern[i * n + j]
    
    return (w * scalar) / num_patterns

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
num_patterns = parsed_toml['simulation_parameters']['num_patterns']

not_unique = True
too_correlated = True
while not_unique or too_correlated:
    patterns = []
    for i in range(num_patterns):
        p = np.random.binomial(1, p_on, num)
        # p = p * 2 - 1

        patterns.append(p)

    not_unique = check_uniqueness(patterns)    
    too_correlated = calculate_correlation(np.array(patterns) / num).sum() > parsed_toml['simulation_parameters']['correlation_threshold']

w = get_weights(num, patterns, a=parsed_toml['simulation_parameters']['a'], b=parsed_toml['simulation_parameters']['b'], scalar=parsed_toml['simulation_parameters']['weights_scalar'] / num_patterns)
w_ie = weights_ie(exc_n, parsed_toml['simulation_parameters']['inh_weights_scalar'], patterns, num_patterns)

def skewed_random(x, y, skew_factor=1, size=1):
    rand = np.random.beta(skew_factor, 1, size=size)
    
    return x + rand * (y - x)

def setup_neuron(neuron):
    neuron.current_voltage = skewed_random(-65, 30, 0.1)[0]
    neuron.c_m = 25

    return neuron

def reset_spike_train(neuron):
    neuron.chance_of_firing = 0

    return neuron

def get_spike_train_setup_function(pattern_index, distortion, firing_rate, stay_unflipped=False):
    def setup_spike_train(pos, neuron):
        x, y = pos
        index = x * exc_n + y
        state = patterns[pattern_index][index] == 1

        if np.random.uniform(0, 1) < distortion:
            if not stay_unflipped:
                state ^= 1
            else:
                if state != 0:
                    state = 0

        if state:
            neuron.chance_of_firing = firing_rate
        else:
            neuron.chance_of_firing = 0

        return neuron

    return setup_spike_train

def get_spike_train_same_firing_rate_setup(firing_rate):
    def setup_spike_train(neuron):
        neuron.chance_of_firing = firing_rate
    
        return neuron

    return setup_spike_train

def get_noisy_spike_train_setup_function(noise_level, firing_rate):
    def setup_spike_train(neuron):
        if np.random.uniform(0, 1) < noise_level:
            neuron.chance_of_firing = firing_rate
        else:
            neuron.chance_of_firing = 0
        
        return neuron

    return setup_spike_train

def find_peaks_above_threshold(series, threshold):
    peaks, _ = scipy.signal.find_peaks(np.array(series))
    filtered_peaks = [index for index in peaks if series[index] > threshold]
    
    return filtered_peaks

def try_max(a):
    if len(a) == 0:
        return 0
    else:
        return max(a)

def acc(true_pattern, pred_pattern, threshold=10): 
    current_pred_pattern = pred_pattern
    current_pred_pattern[pred_pattern < threshold] = 0 
    current_pred_pattern[pred_pattern >= threshold] = 1
    return (true_pattern.reshape(exc_n, exc_n) == current_pred_pattern.reshape(exc_n, exc_n)).sum() / (num)

def signal_to_noise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

# keys should just be list(parsed_toml['variables'].keys())
# simplify expr after
combinations = list(itertools.product(*[i for i in parsed_toml['variables'].values()]))

all_states = [dict(zip(list(parsed_toml['variables'].keys(), combination))) for combination in combinations]

# accuracy should check against the main input and the bayesian cue

for current_state in tqdm(all_states):
    for trial in range(parsed_toml['simulation_parameters']['trials']):
        pattern1, pattern2 = np.random.choice(range(num_patterns), 2, replace=False)

        glu_neuro = ln.ApproximateNeurotransmitter(clearance_constant=current_state['glutamate_clearance'])
        gaba_neuro = ln.ApproximateNeurotransmitter(clearance_constant=current_state['gabaa_clearance'])

        exc_neurotransmitters = ln.DopaGluGABAApproximateNeurotransmitters()
        exc_neurotransmitters.set_neurotransmitter(ln.DopaGluGABANeurotransmitterType.Glutamate, glu_neuro)

        inh_neurotransmitters = ln.DopaGluGABAApproximateNeurotransmitters()
        inh_neurotransmitters.set_neurotransmitter(ln.DopaGluGABANeurotransmitterType.GABA, gaba_neuro)

        glu = ln.GlutamateReceptor()
        gabaa = ln.GABAReceptor()

        glu.ampa_g = current_state['nmda_g']
        glu.nmda_g = current_state['ampa_g']
        gabaa.g = current_state['gabaa_g']

        poisson = ln.DopaPoissonNeuron()
        poisson.set_neurotransmitters(exc_neurotransmitters)

        inh_lattice = ln.DopaIzhikevichLattice(0)
        inh_lattice.populate(inh_neuron, inh_n, inh_n)
        inh_lattice.apply(setup_neuron)

        exc_lattice = ln.DopaIzhikevichLattice(1)
        exc_lattice.populate(exc_neuron, exc_n, exc_n)
        exc_lattice.apply(setup_neuron)
        position_to_index = exc_lattice.position_to_index
        exc_lattice.connect(
            lambda x, y: bool(w[position_to_index[x]][position_to_index[y]] != 0), 
            lambda x, y: w[position_to_index[x]][position_to_index[y]],
        )
        exc_lattice.update_grid_history = True

        spike_train_lattice = ln.DopaPoissonLattice(2)
        spike_train_lattice.populate(poisson, exc_n, exc_n)

        cue_lattice = ln.DopaPoissonLattice(3)
        cue_lattice.populate(poisson, 1, 1)

        network = ln.DopaIzhikevichNetwork.generate_network([exc_lattice, inh_lattice], [spike_train_lattice, cue_lattice])
        network.connect(
            0, 1, 
            lambda x, y: True, 
            lambda x, y: w_ie[int(position_to_index[y] / exc_n), position_to_index[y] % exc_n],
        )
        network.connect(
            1, 0, 
            lambda x, y: np.random.uniform() <= current_state['prob_of_exc_to_inh'], 
            lambda x, y: current_state['exc_to_inh'],
        )
        network.connect(2, 1, lambda x, y: x == y, lambda x, y: current_state['spike_train_to_exc'])
        network.connect(3, 1, lambda x, y: bool(patterns[pattern2][y[0] * exc_n + y[1]] == 1), lambda x, y: current_state['bayesian_to_exc'])

        network.set_dt(1)
        network.parallel = True

        network.apply_spike_train_lattice_given_position(
            2, 
            get_spike_train_setup_function(
                pattern1, 
                current_state['distortion'],
                parse_toml['simulation_parameters']['main_firing_rate'],
                True,
            )
        )

        network.apply_spike_train_lattice(
            3, 
            get_spike_train_same_firing_rate_setup(
                parse_toml['simulation_parameters']['bayesian_firing_rate'],
            )
        )

        # check accuracy on bayesian pattern and main pattern
        