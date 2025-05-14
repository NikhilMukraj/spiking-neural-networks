# bayesian inference using a single on/off cue
# bayesian inference using another memory as a cue
# bayesian inference using d1/d2
# option to enable d1 or d2 on exc or inh group

import toml
import json
import sys
import itertools
import numpy as np
from tqdm import tqdm
from pipeline_setup import parse_toml, generate_key_helper
from pipeline_setup import get_weights, weights_ie, check_uniqueness, generate_patterns
from pipeline_setup import calculate_correlation, skewed_random, generate_setup_neuron
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

    if 'pattern_switch' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['pattern_switch'] = False
    
    if 'memory_biases_memory' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['memory_biases_memory'] = False

    if 'main_noisy' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['main_noisy'] = False
    if 'noisy_cue_noise_level' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['noisy_cue_noise_level'] = 0.1

    if 'bayesian_1_on' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['bayesian_1_on'] = True
    if 'bayesian_2_on' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['bayesian_2_on'] = True
    if 'main_1_on' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['main_1_on'] = True
    if 'main_2_on' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['main_2_on'] = True

    if 'd1' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['d1'] = False
    if 'd2' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['d2'] = False
    
    if parsed['simulation_parameters']['d1'] and parsed['simulation_parameters']['d2']:
        raise ValueError('D1 and D2 cannot both be active, must be one or the other or neither')

    if 'peaks_on' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['peaks_on'] = False
    
    if 'measure_snr' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['measure_snr'] = False

    if 'distortion_on_only' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['distortion_on_only'] = False

    if 'd_acts_on_inh' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['d_acts_on_inh'] = False
    
    if 'first_window' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['first_window'] = 1_000
    if 'second_window' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['second_window'] = 1_000

    if 'trials' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['trials'] = 10
    if 'gpu_batch' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['gpu_batch'] = 10
    if parsed['simulation_parameters']['trials'] % parsed['simulation_parameters']['gpu_batch'] != 0:
        raise ValueError('`trials` must be visible by `gpu_batch` size')

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

    if 'reset_patterns' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['reset_patterns'] = False

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

    if 'prob_of_d_to_inh' not in parsed['variables']:
        parsed['variables']['prob_of_d_to_inh'] = [1]

    if 'nmda_g' not in parsed['variables']:
        parsed['variables']['nmda_g'] = [0.6]
    if 'ampa_g' not in parsed['variables']:
        parsed['variables']['ampa_g'] = [1]
    if 'gabaa_g' not in parsed['variables']:
        parsed['variables']['gabaa_g'] = [1.2]

    if 's_d1' not in parsed['variables']:
        parsed['variables']['s_d1'] = [0] # [1]
    if 's_d2' not in parsed['variables']:
        parsed['variables']['s_d2'] = [0] # [0.025]

    if 'glutamate_clearance' not in parsed['variables']:
        parsed['variables']['glutamate_clearance'] = [0.001]
    if 'gabaa_clearance' not in parsed['variables']:
        parsed['variables']['gabaa_clearance'] = [0.001]
    if 'dopamine_clearance' not in parsed['variables']:
        parsed['variables']['dopamine_clearance'] = [0.001]

    # single on/off cue versus entire group

fields = [
    'main_firing_rate', 'bayesian_firing_rate', 'distortion', 'bayesian_distortion',
    'prob_of_exc_to_inh', 'exc_to_inh', 'spike_train_to_exc', 'bayesian_to_exc',
    'prob_of_d_to_inh',
    'nmda_g', 'ampa_g', 'gabaa_g', 's_d1', 's_d2',
    'glutamate_clearance', 'gabaa_clearance', 'dopamine_clearance',
]

def generate_key(parsed, current_state):
    key = []

    key.append(f'trial: {current_state["trial"]}')

    key.append(f'pattern1: {current_state["pattern1"]}')
    key.append(f'pattern2: {current_state["pattern2"]}')

    if 'switched_pattern' in current_state:
        key.append(f'switched_pattern: {current_state["switched_pattern"]}')
    
    for field in fields:
        generate_key_helper(current_state, key, parsed, field)

    return ', '.join(key)

with open(sys.argv[1], 'r') as f:
    parsed_toml = parse_toml(f)

fill_defaults(parsed_toml)

if any(i not in fields for i in parsed_toml['variables']):
    raise ValueError(f'Unkown variables: {[i for i in parsed_toml["variables"] if i not in fields]}')

exc_n = parsed_toml['simulation_parameters']['exc_n']
num = exc_n * exc_n

inh_n = parsed_toml['simulation_parameters']['inh_n']

setup_neuron = generate_setup_neuron(
    parsed_toml['simulation_parameters']['c_m'], 
    parsed_toml['simulation_parameters']['skew'],
)

p_on = 0.5
num_patterns = parsed_toml['simulation_parameters']['num_patterns']

patterns = generate_patterns(num, p_on, num_patterns, parsed_toml['simulation_parameters']['correlation_threshold'])
if parsed_toml['simulation_parameters']['memory_biases_memory']:
    bayesian_memory_patterns = generate_patterns(num, p_on, num_patterns, parsed_toml['simulation_parameters']['correlation_threshold'])

w = get_weights(num, patterns, a=parsed_toml['simulation_parameters']['a'], b=parsed_toml['simulation_parameters']['b'], scalar=parsed_toml['simulation_parameters']['weights_scalar'] / num_patterns)
w_ie = weights_ie(exc_n, parsed_toml['simulation_parameters']['inh_weights_scalar'], patterns, num_patterns)

combinations = list(itertools.product(*[i for i in parsed_toml['variables'].values()]))

all_states = [dict(zip(list(parsed_toml['variables'].keys()), combination)) for combination in combinations]


np.seterr(divide='ignore', invalid='ignore')

simulation_output = {}

i1 = 0
e1 = 1
c1 = 2
c2 = 3
i2 = 4
e2 = 5
d = 6

all_ids = [i1, e1, c1, c2, i2, e2, d]
network_batch_size = len(all_ids)

# compile kernels beforehand
network = ln.IzhikevichNeuronNetworkGPU()

print(json.dumps(parsed_toml, indent=4))

for current_state in tqdm(all_states):
    glu_neuro = ln.BoundedNeurotransmitterKinetics(clearance_constant=current_state['glutamate_clearance'], t_max=10)
    gaba_neuro = ln.BoundedNeurotransmitterKinetics(clearance_constant=current_state['gabaa_clearance'])
    dopa_neuro = ln.BoundedNeurotransmitterKinetics(clearance_constant=current_state['dopamine_clearance'])

    exc_neurotransmitters = {ln.DopaGluGABANeurotransmitterType.Glutamate: glu_neuro}
    inh_neurotransmitters = {ln.DopaGluGABANeurotransmitterType.GABA: gaba_neuro}
    dopa_neurotransmitters = {ln.DopaGluGABANeurotransmitterType.Dopamine: dopa_neuro}

    glu = ln.GlutamateReceptor(ampa_r=ln.BoundedReceptorKinetics(r_max=10), nmda_r=ln.BoundedReceptorKinetics(r_max=10))
    gabaa = ln.GABAReceptor()
    dopamine_rs = ln.DopamineReceptor()

    dopamine_rs.s_d1 = current_state['s_d1']
    dopamine_rs.s_d2 = current_state['s_d2']

    glu.g_ampa = current_state['nmda_g']
    glu.g_nmda = current_state['ampa_g']
    gabaa.g = current_state['gabaa_g']

    poisson = ln.PoissonNeuron()
    poisson.set_synaptic_neurotransmitters(exc_neurotransmitters)

    receptors = ln.DopaGluGABA()
    receptors.insert(ln.DopaGluGABANeurotransmitterType.Glutamate, glu)
    receptors.insert(ln.DopaGluGABANeurotransmitterType.GABA, gabaa)
    receptors.insert(ln.DopaGluGABANeurotransmitterType.Dopamine, dopamine_rs)

    exc_neuron = ln.IzhikevichNeuron()
    exc_neuron.set_synaptic_neurotransmitters(exc_neurotransmitters)
    exc_neuron.set_receptors(receptors)

    inh_neuron = ln.IzhikevichNeuron()
    inh_neuron.set_synaptic_neurotransmitters(inh_neurotransmitters)
    inh_neuron.set_receptors(receptors)

    # determine how many trials to run in parallel
    # add them to network
    # connect accordingly
    # record data

    for trial_batch in range(int(parsed_toml['simulation_parameters']['trials'] / parsed_toml['simulation_parameters']['gpu_batch'])):
        all_patterns = []
        all_ws = []
        all_w_ies = []
        all_position_to_indexes = []

        all_pattern1s = []
        all_pattern2s = []

        i1s = []
        e1s = []
        c1s = []
        c2s = []

        all_first_accs = []
        all_bayesian_first_accs = []

        for element in range(parsed_toml['simulation_parameters']['gpu_batch']):
            if parsed_toml['simulation_parameters']['reset_patterns']:
                patterns = generate_patterns(num, p_on, num_patterns, parsed_toml['simulation_parameters']['correlation_threshold'])
                if parsed_toml['simulation_parameters']['memory_biases_memory']:
                    bayesian_memory_patterns = generate_patterns(num, p_on, num_patterns, parsed_toml['simulation_parameters']['correlation_threshold'])
                    all_bayesian_memory_patterns.append(bayesian_memory_patterns)           

                w = get_weights(num, patterns, a=parsed_toml['simulation_parameters']['a'], b=parsed_toml['simulation_parameters']['b'], scalar=parsed_toml['simulation_parameters']['weights_scalar'] / num_patterns)
                w_ie = weights_ie(exc_n, parsed_toml['simulation_parameters']['inh_weights_scalar'], patterns, num_patterns)

            all_patterns.append(patterns)
            all_ws.append(w)
            all_w_ies.append(w_ie)

            if parsed_toml['simulation_parameters']['bayesian_is_not_main']:
                pattern1, pattern2 = np.random.choice(range(num_patterns), 2, replace=False)
            else:
                pattern1 = np.random.choice(range(num_patterns))
                pattern2 = pattern1

            all_pattern1s.append(pattern1)
            all_pattern2s.append(pattern2)

            inh_lattice = ln.IzhikevichNeuronLattice(element * network_batch_size + i1)
            inh_lattice.populate(inh_neuron, inh_n, inh_n)
            inh_lattice.apply(setup_neuron)

            exc_lattice = ln.IzhikevichNeuronLattice(element * network_batch_size + e1)
            exc_lattice.populate(exc_neuron, exc_n, exc_n)
            exc_lattice.apply(setup_neuron)
            position_to_index = exc_lattice.position_to_index
            exc_lattice.connect(
                lambda x, y: bool(w[position_to_index[x]][position_to_index[y]] != 0), 
                lambda x, y: w[position_to_index[x]][position_to_index[y]],
            )
            exc_lattice.update_grid_history = True

            spike_train_lattice = ln.PoissonLattice(element * network_batch_size + c1)
            spike_train_lattice.populate(poisson, exc_n, exc_n)

            all_position_to_indexes.append(position_to_index)

            i1s.append(inh_lattice)
            e1s.append(exc_lattice)
            c1s.append(spike_train_lattice)

            if parsed_toml['simulation_parameters']['d1'] or parsed_toml['simulation_parameters']['d2']:
                poisson_dopa = ln.PoissonNeuron()
                poisson_dopa.set_synaptic_neurotransmitters(dopa_neurotransmitters)

                cue_lattice = ln.PoissonLattice(element * network_batch_size + c2)
                cue_lattice.populate(poisson_dopa, exc_n, exc_n)
            else:
                cue_lattice = ln.PoissonLattice(element * network_batch_size + c2)
                cue_lattice.populate(poisson, exc_n, exc_n)

            c2s.append(cue_lattice)
    
        [network.add_lattice(i) for i in i1s]
        [network.add_lattice(i) for i in e1s]
        [network.add_spike_train_lattice(i) for i in c1s]
        [network.add_spike_train_lattice(i) for i in c2s]

        # print(f'i1: {[i.id for i in i1s]}')
        # print(f'e1: {[i.id for i in e1s]}')
        # print(f'c1: {[i.id for i in c1s]}')
        # print(f'i2: {[i.id for i in i2s]}')
        # print(f'e2: {[i.id for i in e2s]}')
        # print(f'c2: {[i.id for i in c2s]}')
        # print(f'd: {[i.id for i in ds]}')

        for element in range(parsed_toml['simulation_parameters']['gpu_batch']):
            network.connect(
                element * network_batch_size + i1, element * network_batch_size + e1, 
                lambda x, y: True, 
                lambda x, y: all_w_ies[element][int(all_position_to_indexes[element][y] / exc_n), all_position_to_indexes[element][y] % exc_n],
            )
            network.connect(
                element * network_batch_size + e1, element * network_batch_size + i1, 
                lambda x, y: np.random.uniform() <= current_state['prob_of_exc_to_inh'], 
                lambda x, y: current_state['exc_to_inh'],
            )
            network.connect(
                element * network_batch_size + c1, 
                element * network_batch_size + e1, 
                lambda x, y: x == y, 
                lambda x, y: current_state['spike_train_to_exc']
            )
            network.connect(
                element * network_batch_size + c2, 
                element * network_batch_size + e1, 
                lambda x, y: x == y, 
                lambda x, y: current_state['bayesian_to_exc']
            )

        network.set_dt(parsed_toml['simulation_parameters']['dt'])

        network.electrical_synapse = False
        network.chemical_synapse = True

        if parsed_toml['simulation_parameters']['main_1_on']:
            main_firing_rate = current_state['main_firing_rate']
        else:
            main_firing_rate = 0

        for element in range(parsed_toml['simulation_parameters']['gpu_batch']):
            if not parsed_toml['simulation_parameters']['main_noisy']:
                network.apply_spike_train_lattice_given_position(
                    element * network_batch_size + c1, 
                    get_spike_train_setup_function(
                        all_patterns[element],
                        all_pattern1s[element], 
                        current_state['distortion'],
                        main_firing_rate,
                        exc_n,
                        parsed_toml['simulation_parameters']['distortion_on_only'],
                    )
                )
            else:
                network.apply_spike_train_lattice(
                    element * network_batch_size + c1, 
                    get_noisy_spike_train_setup_function(
                        parsed_toml['simulation_parameters']['noisy_cue_noise_level'],
                        main_firing_rate,
                    )
                )

        if parsed_toml['simulation_parameters']['bayesian_1_on']:
            bayesian_firing_rate = current_state['bayesian_firing_rate']
        else:
            bayesian_firing_rate = 0

        for element in range(parsed_toml['simulation_parameters']['gpu_batch']):
            if parsed_toml['simulation_parameters']['d1']:
                network.apply_spike_train_lattice_given_position(
                    element * network_batch_size + c2, 
                    get_spike_train_setup_function(
                        all_patterns[element],
                        all_pattern2s[element], 
                        current_state['bayesian_distortion'],
                        bayesian_firing_rate,
                        exc_n,
                        parsed_toml['simulation_parameters']['distortion_on_only'],
                    )
                )
            elif parsed_toml['simulation_parameters']['d2']:
                inv_patterns = [np.logical_not(i).astype(int) for i in all_patterns[element]]

                network.apply_spike_train_lattice_given_position(
                    element * network_batch_size + c2, 
                    get_spike_train_setup_function(
                        inv_patterns,
                        all_pattern2s[element], 
                        current_state['bayesian_distortion'],
                        bayesian_firing_rate,
                        exc_n,
                        parsed_toml['simulation_parameters']['distortion_on_only'],
                    )
                )

        network.run_lattices(parsed_toml['simulation_parameters']['iterations1'])

        for element in range(parsed_toml['simulation_parameters']['gpu_batch']):
            hist = network.get_lattice(element * network_batch_size + e1).history
            data = [i.flatten() for i in np.array(hist)]
            peaks = [find_peaks_above_threshold([j[i] for j in data], 20) for i in range(len(data[0]))]

            first_window = parsed_toml['simulation_parameters']['iterations1'] - parsed_toml['simulation_parameters']['first_window'] 

            current_pred_pattern = np.array([len([j for j in i if j >= first_window]) for i in peaks])
            firing_max = current_pred_pattern.max()

            first_acc = determine_accuracy(
                all_patterns[element],
                all_pattern1s[element],
                num_patterns,
                first_window,
                peaks,
                exc_n,
                parsed_toml['simulation_parameters']['use_correlation_as_accuracy'],
                parsed_toml['simulation_parameters']['get_all_accuracies'],
            )

            all_first_accs.append(first_acc)

            if parsed_toml['simulation_parameters']['bayesian_is_not_main']:
                bayesian_first_acc = determine_accuracy(
                    all_patterns[element],
                    all_pattern2s[element],
                    num_patterns,
                    first_window,
                    peaks,
                    exc_n,
                    parsed_toml['simulation_parameters']['use_correlation_as_accuracy'],
                    parsed_toml['simulation_parameters']['get_all_accuracies'],
                )

                all_bayesian_first_accs.append(bayesian_first_acc)

            current_state['trial'] = trial_batch * parsed_toml['simulation_parameters']['gpu_batch'] + element
            current_state['pattern1'] = all_pattern1s[element]
            current_state['pattern2'] = all_pattern2s[element]
            if parsed_toml['simulation_parameters']['pattern_switch']:
                current_state['switched_pattern'] = all_pattern_switches[element]

            key = generate_key(parsed_toml, current_state)

            current_value = {}
            current_value['first_acc'] = all_first_accs[element]
            if parsed_toml['simulation_parameters']['memory_biases_memory']:
                current_value['memory_biases_memory_first_acc'] = all_bayesian_memory_first_accs[element]

            if parsed_toml['simulation_parameters']['bayesian_is_not_main']:
                current_value['bayesian_first_acc'] = all_bayesian_first_accs[element]

            if parsed_toml['simulation_parameters']['measure_snr']:
                signal = np.array([np.array(i).mean() for i in network.get_lattice(element * network_batch_size + e1).history])

                current_value['first_snr'] = float(signal_to_noise(signal[:parsed_toml['simulation_parameters']['iterations1']]))

            if parsed_toml['simulation_parameters']['peaks_on']:
                current_value['peaks'] = [[int(item) for item in sublist] for sublist in peaks]

            simulation_output[key] = current_value

        network.clear()

with open(parsed_toml['simulation_parameters']['filename'], 'w') as file:
    json.dump(simulation_output, file, indent=4)

print("\033[92mFinished simulation\033[0m")
