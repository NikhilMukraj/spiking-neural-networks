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
from pipeline_setup import parse_toml, try_max
from pipeline_setup import get_weights, weights_ie, check_uniqueness
from pipeline_setup import calculate_correlation, skewed_random, setup_neuron
from pipeline_setup import reset_spike_train, get_spike_train_setup_function
from pipeline_setup import get_spike_train_same_firing_rate_setup, get_noisy_spike_train_setup_function
from pieline_setup import find_peaks_above_threshold, acc, signal_to_noise, determine_accuracy
import lixirnet as ln


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

    if 'bayesian_1_on' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['bayesian_1_on'] = True
    if 'bayesian_2_on' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['bayesian_2_on'] = True
    if 'main_1_on' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['main_1_on'] = True
    if 'main_2_on' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['main_2_on'] = True

    if 'peaks_on' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['peaks_on'] = False
    
    if 'measure_snr' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['measure_snr'] = False

    if 'distortion_on_only' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['distortion_on_only'] = False
    
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

    if 'main_firing_rate' not in parsed['variables']:
        parsed['variables']['main_firing_rate'] = [0.01]
    if 'bayesian_firing_rate' not in parsed['simulation_parameters']:
        parsed['variables']['bayesian_firing_rate'] = [0.01]

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

def generate_key(parsed, current_state):
    key = []

    key.append(f'trial: {current_state["trial"]}')

    key.append(f'pattern1: {current_state["pattern1"]}')
    key.append(f'pattern2: {current_state["pattern2"]}')

    generate_key_helper(key, parsed, 'main_firing_rate')
    generate_key_helper(key, parsed, 'bayesian_firing_rate')

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

with open(sys.argv[1], 'r') as f:
    parsed_toml = parse_toml(f)

fill_defaults(parsed_toml)

exc_n = parsed_toml['simulation_parameters']['exc_n']
num = exc_n * exc_n

inh_n = parsed_toml['simulation_parameters']['inh_n']

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

combinations = list(itertools.product(*[i for i in parsed_toml['variables'].values()]))

all_states = [dict(zip(list(parsed_toml['variables'].keys(), combination))) for combination in combinations]

# accuracy should check against the main input and the bayesian cue

for current_state in tqdm(all_states):
    for trial in range(parsed_toml['simulation_parameters']['trials']):
        if parsed_toml['simulation_parameters']['bayesian_is_not_main']:
            pattern1, pattern2 = np.random.choice(range(num_patterns), 2, replace=False)
        else:
            pattern1 = np.random.choice(range(num_patterns))
            pattern2 = pattern1

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

        network.electrical_synapse = False
        network.chemical_synapse = True

        if parsed_toml['main_1_on']:
            main_firing_rate = current_state['main_firing_rate']
        else:
            main_firing_rate = 0

        network.apply_spike_train_lattice_given_position(
            2, 
            get_spike_train_setup_function(
                pattern1, 
                current_state['distortion'],
                main_firing_rate,
                parse_toml['simulation_parameters']['distortion_on_only'],
            )
        )

        if parsed_toml['bayesian_1_on']:
            bayesian_firing_rate = current_state['bayesian_firing_rate']
        else:
            bayesian_firing_rate = 0

        network.apply_spike_train_lattice(
            3, 
            get_spike_train_same_firing_rate_setup(
                bayesian_firing_rate,
            )
        )

        for _ in range(parse_toml['simulation_parameters']['iterations1']):
            network.run_lattices(1)

        hist = network.get_lattice(1).history
        data = [i.flatten() for i in np.array(hist)]
        peaks = [find_peaks_above_threshold([j[i] for j in data], 20) for i in range(len(data[0]))]

        first_window = parsed_toml['simulation_parameters']['iterations1'] - parsed_toml['simulation_parameters']['first_window'] 

        current_pred_pattern = np.array([len([j for j in i if j >= first_window]) for i in peaks])
        firing_max = current_pred_pattern.max()

        first_acc = determine_accuracy(
            pattern2,
            num_patterns,
            first_window,
            peaks,
            parsed_toml['simulation_parameters']['use_correlation_as_accuracy'],
            parsed_toml['simulation_parameters']['get_all_accuracies'],
        )

        if parsed_toml['main_2_on']:
            main_firing_rate = current_state['main_firing_rate']
        else:
            main_firing_rate = 0

        network.apply_spike_train_lattice_given_position(
            2, 
            get_spike_train_setup_function(
                pattern1, 
                current_state['distortion'],
                main_firing_rate,
                parse_toml['simulation_parameters']['distortion_on_only'],
            )
        )

        if parsed_toml['bayesian_2_on']:
            bayesian_firing_rate = current_state['bayesian_firing_rate']
        else:
            bayesian_firing_rate = 0

        network.apply_spike_train_lattice(
            3, 
            get_spike_train_same_firing_rate_setup(
                bayesian_firing_rate,
            )
        )

        for _ in range(parsed_toml['simulation_parameters']['iterations2']):
            network.run_lattices(1)

        hist = network.get_lattice(1).history
        data = [i.flatten() for i in np.array(hist)]
        peaks = [find_peaks_above_threshold([j[i] for j in data], 20) for i in range(len(data[0]))]

        second_window = parsed_toml['simulation_parameters']['iterations2'] - parsed_toml['simulation_parameters']['second_window'] 

        current_pred_pattern = np.array([len([j for j in i if j >= second_window]) for i in peaks])
        firing_max = current_pred_pattern.max()

        if parsed_toml['simulation_parameters']['second_cue'] == False:
            pattern2 = pattern1

        if parsed_toml['simulation_parameters']['iterations2'] != 0:
            second_acc = determine_accuracy(
                pattern2,
                num_patterns,
                second_window,
                peaks,
                parsed_toml['simulation_parameters']['use_correlation_as_accuracy'],
                parsed_toml['simulation_parameters']['get_all_accuracies'],
            )
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
            if parsed_toml['simulation_parameters']['iterations2'] != 0:
                current_value['second_snr'] = float(signal_to_noise(signal[parsed_toml['simulation_parameters']['iterations1']:]))
            else:
                current_value['second_snr'] = None

        if parsed_toml['simulation_parameters']['peaks_on']:
            current_value['peaks'] = [[int(item) for item in sublist] for sublist in peaks]

        simulation_output[key] = current_value

        # check accuracy on bayesian pattern and main pattern

with open(parsed_toml['simulation_parameters']['filename'], 'w') as file:
    json.dump(simulation_output, file, indent=4)

print("\033[92mFinished simulation\033[0m")
