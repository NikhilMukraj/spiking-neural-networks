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

    if 'second_cue' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['second_cue'] = True
    if 'second_cue_is_noisy' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['second_cue_is_noisy'] = False
    if 'first_cue_is_noisy' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['first_cue_is_noisy'] = False
    if 'noisy_cue_noise_level' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['noisy_cue_noise_level'] = 0.1

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

    if 'distortion' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['distortion'] = 0.15

    if 'dt' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['dt'] = 1
    if 'c_m' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['c_m'] = 25

    if 'prob_of_exc_to_inh' not in parsed['variables']:
        parsed['variables']['prob_of_exc_to_inh'] = [0.5]
    if 'exc_to_inh' not in parsed['variables']:
        parsed['variables']['exc_to_inh'] = [1]
    if 'spike_train_to_exc' not in parsed['variables']:
        parsed['variables']['spike_train_to_exc'] = [5]

    if 'nmda_g' not in parsed['variables']:
        parsed['variables']['nmda_g'] = [0.6]
    if 'ampa_g' not in parsed['variables']:
        parsed['variables']['ampa_g'] = [1]
    if 'gabaa_g' not in parsed['variables']:
        parsed['variables']['gabaa_g'] = [1.2]

    if 'glutamate_clearance' not in parsed['variables']:
        if 'nmda_clearance' not in parsed['variables']:
            parsed['variables']['nmda_clearance'] = [0.001]
        if 'ampa_clearance' not in parsed['variables']:
            parsed['variables']['ampa_clearance'] = [0.001]
        parsed['simulation_parameters']['use_glutamate_clearance'] = False
    else:
        parsed['variables']['nmda_clearance'] = parsed['variables']['glutamate_clearance']
        parsed['variables']['ampa_clearance'] = parsed['variables']['glutamate_clearance']
        parsed['simulation_parameters']['use_glutamate_clearance'] = True
    if 'gabaa_clearance' not in parsed['variables']:
        parsed['variables']['gabaa_clearance'] = [0.001]

def generate_key_helper(key, parsed, given_key):
    if len(parsed['variables'][given_key]) != 1:
        key.append(f'{given_key}: {current_state[given_key]}')

def generate_key(parsed, current_state):
    key = []

    key.append(f'trial: {current_state["trial"]}')

    key.append(f'pattern1: {current_state["pattern1"]}')
    key.append(f'pattern2: {current_state["pattern2"]}')

    generate_key_helper(key, parsed, 'prob_of_exc_to_inh')
    generate_key_helper(key, parsed, 'exc_to_inh')
    generate_key_helper(key, parsed, 'spike_train_to_exc')

    generate_key_helper(key, parsed, 'nmda_g')
    generate_key_helper(key, parsed, 'ampa_g')
    generate_key_helper(key, parsed, 'gabaa_g')

    generate_key_helper(key, parsed, 'nmda_clearance')
    generate_key_helper(key, parsed, 'ampa_clearance')
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

# w = get_weights(num, patterns, a=1, b=1, scalar=parsed_toml['simulation_parameters']['weights_scalar'] / num_patterns)
w = get_weights(num, patterns, a=parsed_toml['simulation_parameters']['a'], b=parsed_toml['simulation_parameters']['b'], scalar=parsed_toml['simulation_parameters']['weights_scalar'] / num_patterns)
w_ie = weights_ie(exc_n, parsed_toml['simulation_parameters']['inh_weights_scalar'], patterns, num_patterns)

def skewed_random(x, y, skew_factor=1, size=1):
    rand = np.random.beta(skew_factor, 1, size=size)
    
    return x + rand * (y - x)

def setup_neuron(neuron):
    neuron.current_voltage = skewed_random(-65, 30, parsed_toml['simulation_parameters']['skew'])[0]
    neuron.c_m = parsed_toml['simulation_parameters']['c_m']

    return neuron

def reset_spike_train(neuron):
    neuron.chance_of_firing = 0

    return neuron

def get_spike_train_setup_function(pattern_index, distortion):
    def setup_spike_train(pos, neuron):
        x, y = pos
        index = x * exc_n + y
        state = patterns[pattern_index][index] == 1

        if np.random.uniform(0, 1) < distortion:
            state ^= 1

        if state:
            neuron.chance_of_firing = 0.001
        else:
            neuron.chance_of_firing = 0

        return neuron

    return setup_spike_train

def get_noisy_spike_train_setup_function(noise_level):
    def setup_spike_train(neuron):
        if np.random.uniform(0, 1) < noise_level:
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
    return (true_pattern.reshape(exc_n, exc_n) == current_pred_pattern.reshape(exc_n, exc_n)).sum() / (num)

def signal_to_noise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)

keys = [
    'exc_to_inh', 'prob_of_exc_to_inh', 'spike_train_to_exc',
    'nmda_g', 'ampa_g', 'gabaa_g',
    'nmda_clearance', 'ampa_clearance', 'gabaa_clearance',
]

combinations = list(itertools.product(*[parsed_toml['variables'][key] for key in keys]))

all_states = [dict(zip(keys, combination)) for combination in combinations]

if parsed_toml['simulation_parameters']['use_glutamate_clearance']:
    all_states = [i for i in all_states if i['nmda_clearance'] == i['ampa_clearance']]

print(json.dumps(parsed_toml, indent=4))

simulation_output = {}

for current_state in tqdm(all_states):
    for trial in range(parsed_toml['simulation_parameters']['trials']):
        pattern1, pattern2 = np.random.choice(range(num_patterns), 2, replace=False)

        distortion = parsed_toml['simulation_parameters']['distortion']

        ampa_neuro = ln.ApproximateNeurotransmitter(clearance_constant=current_state['ampa_clearance'])
        nmda_neuro = ln.ApproximateNeurotransmitter(clearance_constant=current_state['nmda_clearance'])
        gabaa_neuro = ln.ApproximateNeurotransmitter(clearance_constant=current_state['gabaa_clearance'])

        exc_neurotransmitters = ln.ApproximateNeurotransmitters()
        exc_neurotransmitters.set_neurotransmitter(ln.IonotropicNeurotransmitterType.AMPA, ampa_neuro)
        exc_neurotransmitters.set_neurotransmitter(ln.IonotropicNeurotransmitterType.NMDA, nmda_neuro)

        inh_neurotransmitters = ln.ApproximateNeurotransmitters()
        inh_neurotransmitters.set_neurotransmitter(ln.IonotropicNeurotransmitterType.GABAa, gabaa_neuro)

        nmda = ln.ApproximateLigandGatedChannel(ln.IonotropicNeurotransmitterType.NMDA)
        ampa = ln.ApproximateLigandGatedChannel(ln.IonotropicNeurotransmitterType.AMPA)
        gabaa = ln.ApproximateLigandGatedChannel(ln.IonotropicNeurotransmitterType.GABAa)

        nmda.g = current_state['nmda_g']
        ampa.g = current_state['ampa_g']
        gabaa.g = current_state['gabaa_g']

        ligand_gates = ln.ApproximateLigandGatedChannels()
        ligand_gates.set_ligand_gate(ln.IonotropicNeurotransmitterType.NMDA, nmda)
        ligand_gates.set_ligand_gate(ln.IonotropicNeurotransmitterType.AMPA, ampa)
        ligand_gates.set_ligand_gate(ln.IonotropicNeurotransmitterType.GABAa, gabaa)

        exc_neuron = ln.IzhikevichNeuron()
        exc_neuron.set_neurotransmitters(exc_neurotransmitters)
        exc_neuron.set_ligand_gates(ligand_gates)

        inh_neuron = ln.IzhikevichNeuron()
        inh_neuron.set_neurotransmitters(inh_neurotransmitters)
        inh_neuron.set_ligand_gates(ligand_gates)

        poisson = ln.PoissonNeuron()
        poisson.set_neurotransmitters(exc_neurotransmitters)

        inh_lattice = ln.IzhikevichLattice(0)
        inh_lattice.populate(inh_neuron, inh_n, inh_n)
        inh_lattice.apply(setup_neuron)
        # inh_lattice.connect(lambda x, y: x != y, lambda x, y: 1)

        exc_lattice = ln.IzhikevichLattice(1)
        exc_lattice.populate(exc_neuron, exc_n, exc_n)
        exc_lattice.apply(setup_neuron)
        position_to_index = exc_lattice.position_to_index
        exc_lattice.connect(
            lambda x, y: bool(w[position_to_index[x]][position_to_index[y]] != 0), 
            lambda x, y: w[position_to_index[x]][position_to_index[y]],
        )
        exc_lattice.update_grid_history = True

        spike_train_lattice = ln.PoissonLattice(2)
        spike_train_lattice.populate(poisson, exc_n, exc_n)
        if not parsed_toml['simulation_parameters']['first_cue_is_noisy']:
            spike_train_lattice.apply_given_position(get_spike_train_setup_function(pattern1, distortion))
        else:
            spike_train_lattice.apply(get_noisy_spike_train_setup_function(parsed_toml['simulation_parameters']['noisy_cue_noise_level']))

        network = ln.IzhikevichNetwork.generate_network([exc_lattice, inh_lattice], [spike_train_lattice])
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
        network.set_dt(parsed_toml['simulation_parameters']['dt'])
        network.parallel = True

        network.electrical_synapse = False
        network.chemical_synapse = True

        for _ in range(parsed_toml['simulation_parameters']['iterations1']):
            network.run_lattices(1)

        hist = network.get_lattice(1).history
        data = [i.flatten() for i in np.array(hist)]
        peaks = [find_peaks_above_threshold([j[i] for j in data], 20) for i in range(len(data[0]))]

        first_window = parsed_toml['simulation_parameters']['iterations1'] - parsed_toml['simulation_parameters']['first_window'] 

        current_pred_pattern = np.array([len([j for j in i if j >= first_window]) for i in peaks])
        firing_max = current_pred_pattern.max()

        if not parsed_toml['simulation_parameters']['use_correlation_as_accuracy']:
            if not parsed_toml['simulation_parameters']['get_all_accuracies']:
                first_acc = try_max(
                    [acc(patterns[pattern1], np.array([len([j for j in i if j >= first_window]) for i in peaks]), threshold=i) for i in range(0, firing_max)]
                )
                first_acc_inv = try_max(
                    [acc(np.logical_not(patterns[pattern1]).astype(int), np.array([len([j for j in i if j >= first_window]) for i in peaks]), threshold=i) for i in range(0, firing_max)]
                )

                first_acc = max(first_acc, first_acc_inv)
            else:
                accs = []
                for pattern_index in range(num_patterns):
                    first_acc = try_max(
                        [
                            acc(
                                patterns[pattern_index], 
                                np.array([len([j for j in i if j >= first_window]) for i in peaks]), 
                                threshold=i
                            ) for i in range(0, firing_max)
                        ]
                    )

                    inv_acc = try_max(
                        [
                            acc(
                                np.logical_not(patterns[pattern_index]).astype(int), 
                                np.array([len([j for j in i if j >= first_window]) for i in peaks]), 
                                threshold=i
                            ) for i in range(0, firing_max)
                        ]
                    )

                    accs.append(max(first_acc, inv_acc))

                first_acc = [float(i) for i in accs]
        else:
            correlation_coefficients = []
            for pattern_index in range(num_patterns):
                correlation_coefficients.append(
                    np.corrcoef(patterns[pattern_index], np.array([len([j for j in i if j >= first_window]) for i in peaks]))[0, 1]
                )
                
            first_acc = bool(pattern1 == np.argmax(correlation_coefficients))

        if not parsed_toml['simulation_parameters']['second_cue_is_noisy']:
            if parsed_toml['simulation_parameters']['second_cue']:
                network.apply_spike_train_lattice_given_position(2, get_spike_train_setup_function(pattern2, distortion))
            else:
                network.apply_spike_train_lattice(2, reset_spike_train)
        else:
            spike_train_lattice.apply(get_noisy_spike_train_setup_function(parsed_toml['simulation_parameters']['noisy_cue_noise_level']))

        for _ in range(parsed_toml['simulation_parameters']['iterations2']):
            network.run_lattices(1)

        hist = network.get_lattice(1).history
        data = [i.flatten() for i in np.array(hist)]
        peaks = [find_peaks_above_threshold([j[i] for j in data], 20) for i in range(len(data[0]))]

        second_window = parsed_toml['simulation_parameters']['iterations2'] - parsed_toml['simulation_parameters']['second_window'] 

        current_pred_pattern = np.array([len([j for j in i if j >= second_window]) for i in peaks])
        firing_max = current_pred_pattern.max()

        if parsed_toml['simulation_parameters']['iterations2'] != 0:
            if not parsed_toml['simulation_parameters']['use_correlation_as_accuracy']:
                if not parsed_toml['simulation_parameters']['get_all_accuracies']:
                    second_acc = try_max(
                        [acc(patterns[pattern2], np.array([len([j for j in i if j >= second_window]) for i in peaks]), threshold=i) for i in range(0, firing_max)]
                    )
                    second_acc_inv = try_max(
                        [acc(np.logical_not(patterns[pattern2]).astype(int), np.array([len([j for j in i if j >= second_window]) for i in peaks]), threshold=i) for i in range(0, firing_max)]
                    )

                    second_acc = max(second_acc, second_acc_inv)
                else:
                    accs = []
                    for pattern_index in range(num_patterns):
                        second_acc = try_max(
                            [
                                acc(
                                    patterns[pattern_index], 
                                    np.array([len([j for j in i if j >= second_window]) for i in peaks]), 
                                    threshold=i
                                ) 
                                for i in range(0, firing_max)
                            ]
                        )

                        second_acc_inv = try_max(
                            [
                                acc(
                                    np.logical_not(patterns[pattern_index]).astype(int), 
                                    np.array([len([j for j in i if j >= second_window]) for i in peaks]), 
                                    threshold=i
                                ) 
                                for i in range(0, firing_max)
                            ]
                        )

                        accs.append(max(second_acc, second_acc_inv))

                    second_acc = [float(i) for i in accs]
            else:
                correlation_coefficients = []
                for pattern_index in range(num_patterns):
                    correlation_coefficients.append(
                        np.corrcoef(patterns[pattern_index], np.array([len([j for j in i if j >= second_window]) for i in peaks]))[0, 1]
                    )
                    
                second_acc = bool(pattern2 == np.argmax(correlation_coefficients))
        else:
            second_acc = 0

        current_state['trial'] = trial
        current_state['pattern1'] = pattern1
        current_state['pattern2'] = pattern2

        key = generate_key(parsed_toml, current_state)

        current_value = {}
        current_value['first_acc'] = first_acc
        current_value['second_acc'] = second_acc

        if parsed_toml['simulation_parameters']['measure_snr']:
            signal = np.array([np.array(i).mean() for i in hist])

            current_value['first_snr'] = float(signal_to_noise(signal[:parsed_toml['simulation_parameters']['iterations1']]))
            current_value['second_snr'] = float(signal_to_noise(signal[parsed_toml['simulation_parameters']['iterations1']:]))

        if parsed_toml['simulation_parameters']['peaks_on']:
            current_value['peaks'] = [[int(item) for item in sublist] for sublist in peaks]

        simulation_output[key] = current_value

with open(parsed_toml['simulation_parameters']['filename'], 'w') as file:
    json.dump(simulation_output, file, indent=4)

print("\033[92mFinished simulation\033[0m")

# create key and save acc and peaks
# set neurotransmitters + ligand gates parameters
# trials should be incorporated in loop
# add option to stop spike train input after a point and then continue

# key should be modified depending on output
# second acc not calculated if second cue is off

# maybe a signal handler that processes a progress command
