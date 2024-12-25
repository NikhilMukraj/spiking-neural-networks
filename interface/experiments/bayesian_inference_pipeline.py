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
from pipeline_setup import parse_toml, try_max, generate_key_helper
from pipeline_setup import get_weights, weights_ie, check_uniqueness, generate_patterns
from pipeline_setup import calculate_correlation, skewed_random, setup_neuron
from pipeline_setup import reset_spike_train, get_spike_train_setup_function
from pipeline_setup import get_spike_train_same_firing_rate_setup, get_noisy_spike_train_setup_function
from pipeline_setup import find_peaks_above_threshold, acc, signal_to_noise, determine_accuracy
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

    if 'bayesian_is_not_main' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['bayesian_is_not_main'] = True
    
    if 'memory_biases_memory' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['memory_biases_memory'] = False

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
    if 'bayesian_distortion' not in parsed['variables']:
        parsed['variables']['bayesian_distortion'] = [0]

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

    fields = [
        'main_firing_rate', 'bayesian_firing_rate', 'distortion', 'bayesian_distortion',
        'prob_of_exc_to_inh', 'exc_to_inh', 'spike_train_to_exc', 'bayesian_to_exc',
        'nmda_g', 'ampa_g', 'gabaa_g',
        'glutamate_clearance', 'gabaa_clearance'
    ]
    
    for field in fields:
        generate_key_helper(current_state, key, parsed, field)

    return ', '.join(key)

with open(sys.argv[1], 'r') as f:
    parsed_toml = parse_toml(f)

fill_defaults(parsed_toml)

exc_n = parsed_toml['simulation_parameters']['exc_n']
num = exc_n * exc_n

inh_n = parsed_toml['simulation_parameters']['inh_n']

p_on = 0.5
num_patterns = parsed_toml['simulation_parameters']['num_patterns']

patterns = generate_patterns(num, p_on, num_patterns, parsed_toml['simulation_parameters']['correlation_threshold'])
if parsed_toml['simulation_parameters']['memory_biases_memory']:
    bayesian_memory_patterns = generate_patterns(num, p_on, num_patterns, parsed_toml['simulation_parameters']['correlation_threshold'])

w = get_weights(num, patterns, a=parsed_toml['simulation_parameters']['a'], b=parsed_toml['simulation_parameters']['b'], scalar=parsed_toml['simulation_parameters']['weights_scalar'] / num_patterns)
w_ie = weights_ie(exc_n, parsed_toml['simulation_parameters']['inh_weights_scalar'], patterns, num_patterns)

if parsed_toml['simulation_parameters']['memory_biases_memory']:
    w_2 = get_weights(num, bayesian_memory_patterns, a=parsed_toml['simulation_parameters']['a'], b=parsed_toml['simulation_parameters']['b'], scalar=parsed_toml['simulation_parameters']['weights_scalar'] / num_patterns)
    w_ie_2 = weights_ie(exc_n, parsed_toml['simulation_parameters']['inh_weights_scalar'], bayesian_memory_patterns, num_patterns)

combinations = list(itertools.product(*[i for i in parsed_toml['variables'].values()]))

all_states = [dict(zip(list(parsed_toml['variables'].keys()), combination)) for combination in combinations]

print(json.dumps(parsed_toml, indent=4))

simulation_output = {}

i1 = 0
e1 = 1
c1 = 2
c2 = 3
i2 = 4
e2 = 5

for current_state in tqdm(all_states):
    for trial in range(parsed_toml['simulation_parameters']['trials']):
        if parsed_toml['simulation_parameters']['bayesian_is_not_main']:
            pattern1, pattern2 = np.random.choice(range(num_patterns), 2, replace=False)
        else:
            pattern1 = np.random.choice(range(num_patterns))
            pattern2 = pattern1

        if parsed_toml['simulation_parameters']['memory_biases_memory']:
            bayesian_memory_pattern = np.random.choice(range(num_patterns))

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

        receptors = ln.DopaGluGABAReceptors()
        receptors.set_receptor(ln.DopaGluGABANeurotransmitterType.Glutamate, glu)
        receptors.set_receptor(ln.DopaGluGABANeurotransmitterType.GABA, gabaa)

        exc_neuron = ln.DopaIzhikevichNeuron()
        exc_neuron.set_neurotransmitters(exc_neurotransmitters)
        exc_neuron.set_receptors(receptors)

        inh_neuron = ln.DopaIzhikevichNeuron()
        inh_neuron.set_neurotransmitters(inh_neurotransmitters)
        inh_neuron.set_receptors(receptors)

        inh_lattice = ln.DopaIzhikevichLattice(i1)
        inh_lattice.populate(inh_neuron, inh_n, inh_n)
        inh_lattice.apply(setup_neuron)

        exc_lattice = ln.DopaIzhikevichLattice(e1)
        exc_lattice.populate(exc_neuron, exc_n, exc_n)
        exc_lattice.apply(setup_neuron)
        position_to_index = exc_lattice.position_to_index
        exc_lattice.connect(
            lambda x, y: bool(w[position_to_index[x]][position_to_index[y]] != 0), 
            lambda x, y: w[position_to_index[x]][position_to_index[y]],
        )
        exc_lattice.update_grid_history = True

        spike_train_lattice = ln.DopaPoissonLattice(c1)
        spike_train_lattice.populate(poisson, exc_n, exc_n)

        cue_lattice = ln.DopaPoissonLattice(c2)
        cue_lattice.populate(poisson, exc_n, exc_n)

        if parsed_toml['simulation_parameters']['memory_biases_memory']:
            inh_lattice_2 = ln.DopaIzhikevichLattice(i2)
            inh_lattice_2.populate(inh_neuron, inh_n, inh_n)
            inh_lattice.apply(setup_neuron)

            exc_lattice_2 = ln.DopaIzhikevichLattice(e2)
            exc_lattice_2.populate(exc_neuron, exc_n, exc_n)
            exc_lattice_2.apply(setup_neuron)
            position_to_index_2 = exc_lattice_2.position_to_index
            exc_lattice_2.connect(
                lambda x, y: bool(w_2[position_to_index_2[x]][position_to_index_2[y]] != 0), 
                lambda x, y: w_2[position_to_index_2[x]][position_to_index_2[y]],
            )
            exc_lattice_2.update_grid_history = True

        if parsed_toml['simulation_parameters']['memory_biases_memory']:
            network = ln.DopaIzhikevichNetwork.generate_network(
                [exc_lattice, inh_lattice, exc_lattice_2, inh_lattice_2], 
                [spike_train_lattice, cue_lattice],
            )
        else:
            network = ln.DopaIzhikevichNetwork.generate_network(
                [exc_lattice, inh_lattice], 
                [spike_train_lattice, cue_lattice]
            )

        network.connect(
            i1, e1, 
            lambda x, y: True, 
            lambda x, y: w_ie[int(position_to_index[y] / exc_n), position_to_index[y] % exc_n],
        )
        network.connect(
            e1, i1, 
            lambda x, y: np.random.uniform() <= current_state['prob_of_exc_to_inh'], 
            lambda x, y: current_state['exc_to_inh'],
        )
        network.connect(c1, e1, lambda x, y: x == y, lambda x, y: current_state['spike_train_to_exc'])

        if parsed_toml['simulation_parameters']['memory_biases_memory']:
            network.connect(
                i2, e2, 
                lambda x, y: True, 
                lambda x, y: w_ie_2[int(position_to_index_2[y] / exc_n), position_to_index_2[y] % exc_n],
            )
            network.connect(
                e2, i2, 
                lambda x, y: np.random.uniform() <= current_state['prob_of_exc_to_inh'], 
                lambda x, y: current_state['exc_to_inh'],
            )

            network.connect(c2, e2, lambda x, y: x == y, lambda x, y: current_state['spike_train_to_exc'])

            e2_to_e1_mapping = {}
            current_pointer = -1

            for n1, i in enumerate(bayesian_memory_patterns[bayesian_memory_pattern]):
                if i == 0:
                    continue

                to_iterate = list(enumerate(patterns[pattern2]))[current_pointer+1:]

                if len(to_iterate) == 0:
                    break

                for n2, j in to_iterate:
                    if j == 0:
                        continue

                    current_pointer = n2
                    break

                e2_to_e1_mapping[n1] = current_pointer

            network.connect(
                e2, 
                e1, 
                lambda x, y: bool(
                    x[0] * exc_n + x[1] in e2_to_e1_mapping.keys() and 
                    y[0] * exc_n + y[1] in e2_to_e1_mapping.values()
                ), 
                lambda x, y: current_state['bayesian_to_exc']
            )
        else:
            network.connect(c2, e1, lambda x, y: x == y, lambda x, y: current_state['bayesian_to_exc'])

        network.set_dt(1)
        network.parallel = True

        network.electrical_synapse = False
        network.chemical_synapse = True

        if parsed_toml['simulation_parameters']['main_1_on']:
            main_firing_rate = current_state['main_firing_rate']
        else:
            main_firing_rate = 0

        network.apply_spike_train_lattice_given_position(
            c1, 
            get_spike_train_setup_function(
                patterns,
                pattern1, 
                current_state['distortion'],
                main_firing_rate,
                exc_n,
                parsed_toml['simulation_parameters']['distortion_on_only'],
            )
        )

        if parsed_toml['simulation_parameters']['bayesian_1_on']:
            bayesian_firing_rate = current_state['bayesian_firing_rate']
        else:
            bayesian_firing_rate = 0

        if parsed_toml['simulation_parameters']['memory_biases_memory']:
            network.apply_spike_train_lattice_given_position(
                c2, 
                get_spike_train_setup_function(
                    bayesian_memory_patterns,
                    bayesian_memory_pattern, 
                    current_state['bayesian_distortion'],
                    bayesian_firing_rate,
                    exc_n,
                    parsed_toml['simulation_parameters']['distortion_on_only'],
                )
            )
        else:
            network.apply_spike_train_lattice_given_position(
                c2, 
                get_spike_train_setup_function(
                    patterns,
                    pattern2, 
                    current_state['bayesian_distortion'],
                    bayesian_firing_rate,
                    exc_n,
                    parsed_toml['simulation_parameters']['distortion_on_only'],
                )
            )

        for _ in range(parsed_toml['simulation_parameters']['iterations1']):
            network.run_lattices(1)

        hist = network.get_lattice(e1).history
        data = [i.flatten() for i in np.array(hist)]
        peaks = [find_peaks_above_threshold([j[i] for j in data], 20) for i in range(len(data[0]))]

        first_window = parsed_toml['simulation_parameters']['iterations1'] - parsed_toml['simulation_parameters']['first_window'] 

        current_pred_pattern = np.array([len([j for j in i if j >= first_window]) for i in peaks])
        firing_max = current_pred_pattern.max()

        first_acc = determine_accuracy(
            patterns,
            pattern1,
            num_patterns,
            first_window,
            peaks,
            exc_n,
            parsed_toml['simulation_parameters']['use_correlation_as_accuracy'],
            parsed_toml['simulation_parameters']['get_all_accuracies'],
        )
        if parsed_toml['simulation_parameters']['bayesian_is_not_main']:
            bayesian_first_acc = determine_accuracy(
                patterns,
                pattern2,
                num_patterns,
                first_window,
                peaks,
                exc_n,
                parsed_toml['simulation_parameters']['use_correlation_as_accuracy'],
                parsed_toml['simulation_parameters']['get_all_accuracies'],
            )

        if parsed_toml['simulation_parameters']['memory_biases_memory']:
            hist = network.get_lattice(e2).history
            data = [i.flatten() for i in np.array(hist)]
            peaks = [find_peaks_above_threshold([j[i] for j in data], 20) for i in range(len(data[0]))]

            current_pred_pattern = np.array([len([j for j in i if j >= first_window]) for i in peaks])
            firing_max = current_pred_pattern.max()

            bayesian_memory_first_acc = determine_accuracy(
                bayesian_memory_patterns,
                bayesian_memory_pattern,
                num_patterns,
                first_window,
                peaks,
                exc_n,
                parsed_toml['simulation_parameters']['use_correlation_as_accuracy'],
                parsed_toml['simulation_parameters']['get_all_accuracies'],
            )

        if parsed_toml['simulation_parameters']['main_2_on']:
            main_firing_rate = current_state['main_firing_rate']
        else:
            main_firing_rate = 0

        network.apply_spike_train_lattice_given_position(
            c1, 
            get_spike_train_setup_function(
                patterns,
                pattern1, 
                current_state['distortion'],
                main_firing_rate,
                exc_n,
                parsed_toml['simulation_parameters']['distortion_on_only'],
            )
        )

        if parsed_toml['simulation_parameters']['bayesian_2_on']:
            bayesian_firing_rate = current_state['bayesian_firing_rate']
        else:
            bayesian_firing_rate = 0

        if parsed_toml['simulation_parameters']['memory_biases_memory']:
            network.apply_spike_train_lattice_given_position(
                c2, 
                get_spike_train_setup_function(
                    bayesian_memory_patterns,
                    bayesian_memory_pattern, 
                    current_state['bayesian_distortion'],
                    bayesian_firing_rate,
                    exc_n,
                    parsed_toml['simulation_parameters']['distortion_on_only'],
                )
            )
        else:
            network.apply_spike_train_lattice_given_position(
                c2, 
                get_spike_train_setup_function(
                    patterns,
                    pattern2, 
                    current_state['bayesian_distortion'],
                    bayesian_firing_rate,
                    exc_n,
                    parsed_toml['simulation_parameters']['distortion_on_only'],
                )
            )

        for _ in range(parsed_toml['simulation_parameters']['iterations2']):
            network.run_lattices(1)

        hist = network.get_lattice(e1).history
        data = [i.flatten() for i in np.array(hist)]
        peaks = [find_peaks_above_threshold([j[i] for j in data], 20) for i in range(len(data[0]))]

        second_window = parsed_toml['simulation_parameters']['iterations2'] - parsed_toml['simulation_parameters']['second_window'] 

        current_pred_pattern = np.array([len([j for j in i if j >= second_window]) for i in peaks])
        firing_max = current_pred_pattern.max()

        if parsed_toml['simulation_parameters']['iterations2'] != 0:
            second_acc = determine_accuracy(
                patterns,
                pattern1,
                num_patterns,
                second_window,
                peaks,
                exc_n,
                parsed_toml['simulation_parameters']['use_correlation_as_accuracy'],
                parsed_toml['simulation_parameters']['get_all_accuracies'],
            )
            if parsed_toml['simulation_parameters']['bayesian_is_not_main']:
                bayesian_second_acc = determine_accuracy(
                    patterns,
                    pattern2,
                    num_patterns,
                    first_window,
                    peaks,
                    exc_n,
                    parsed_toml['simulation_parameters']['use_correlation_as_accuracy'],
                    parsed_toml['simulation_parameters']['get_all_accuracies'],
                )
            
            if parsed_toml['simulation_parameters']['memory_biases_memory']:
                hist = network.get_lattice(e2).history
                data = [i.flatten() for i in np.array(hist)]
                peaks = [find_peaks_above_threshold([j[i] for j in data], 20) for i in range(len(data[0]))]

                current_pred_pattern = np.array([len([j for j in i if j >= first_window]) for i in peaks])
                firing_max = current_pred_pattern.max()

                bayesian_memory_second_acc = determine_accuracy(
                    bayesian_memory_patterns,
                    bayesian_memory_pattern,
                    num_patterns,
                    first_window,
                    peaks,
                    exc_n,
                    parsed_toml['simulation_parameters']['use_correlation_as_accuracy'],
                    parsed_toml['simulation_parameters']['get_all_accuracies'],
                )
        else:
            second_acc = 0
            bayesian_second_acc = 0

            bayesian_memory_second_acc = 0

        current_state['trial'] = trial
        current_state['pattern1'] = pattern1
        current_state['pattern2'] = pattern2

        key = generate_key(parsed_toml, current_state)

        current_value = {}
        current_value['first_acc'] = first_acc
        if parsed_toml['simulation_parameters']['memory_biases_memory']:
            current_value['memory_biases_memory_first_acc'] = bayesian_memory_first_acc
        if parsed_toml['simulation_parameters']['iterations2'] != 0:
            current_value['second_acc'] = second_acc

            if parsed_toml['simulation_parameters']['memory_biases_memory']:
                current_value['memory_biases_memory_second_acc'] = bayesian_memory_second_acc

        if parsed_toml['simulation_parameters']['bayesian_is_not_main']:
            current_value['bayesian_first_acc'] = bayesian_first_acc
            if parsed_toml['simulation_parameters']['iterations2'] != 0:
                current_value['bayesian_second_acc'] = bayesian_second_acc

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

with open(parsed_toml['simulation_parameters']['filename'], 'w') as file:
    json.dump(simulation_output, file, indent=4)

print("\033[92mFinished simulation\033[0m")

# calculate second memory accuracy if memory biases memory
# test distortion to make sure that everything is running as intended
