# generate auto-associative attractor
# determine firing rates of each neuron when exposed to each cue at varying distortions 
# for n trials
# use umap 3d embedding to visualize distance between attractor states
# save firing rate data and embedding

import toml
import json
import sys
import itertools
import numpy as np
from tqdm import tqdm
from pipeline_setup import parse_toml, generate_key_helper
from pipeline_setup import get_weights, weights_ie, check_uniqueness, generate_patterns
from pipeline_setup import calculate_correlation, skewed_random, generate_setup_neuron
from pipeline_setup import get_spike_train_setup_function
from pipeline_setup import find_peaks_above_threshold, signal_to_noise
import lixirnet as ln


def fill_defaults(parsed):
    if 'simulation_parameters' not in parsed:
        raise ValueError('Requires `simulation_parameters` table')

    if 'filename' not in parsed['simulation_parameters']:
        raise ValueError('Requires `filename` field in `simulation_parameters`')
    
    if 'variables' not in parsed:
        raise ValueError('Requires `variables` table')

    if 'iterations' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['iterations'] = 5_000

    # if 'pattern_switch' not in parsed['simulation_parameters']:
    #     parsed['simulation_parameters']['pattern_switch'] = False

    if 'voltages_on' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['voltages_on'] = False
    
    if 'measure_snr' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['measure_snr'] = False

    if 'distortion_on_only' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['distortion_on_only'] = False

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

    # if 'use_correlation_as_accuracy' not in parsed['simulation_parameters']:
    #     parsed['simulation_parameters']['use_correlation_as_accuracy'] = False
    # if 'get_all_accuracies' not in parsed['simulation_parameters']:
    #     parsed['simulation_parameters']['get_all_accuracies'] = False

    if 'skew' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['skew'] = 0.1

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

    if 'cue_firing_rate' not in parsed['variables']:
        parsed['variables']['cue_firing_rate'] = [0.01]

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
        parsed['variables']['glutamate_clearance'] = [0.001]
    if 'gabaa_clearance' not in parsed['variables']:
        parsed['variables']['gabaa_clearance'] = [0.001]

fields = [
    'cue_firing_rate', 'distortion', 
    'prob_of_exc_to_inh', 'exc_to_inh', 'spike_train_to_exc',
    'nmda_g', 'ampa_g', 'gabaa_g',
    'glutamate_clearance', 'gabaa_clearance',
]

def generate_key(parsed, current_state):
    key = []

    key.append(f'trial: {current_state["trial"]}')

    key.append(f'pattern: {current_state["pattern"]}')
    
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

w = get_weights(num, patterns, a=parsed_toml['simulation_parameters']['a'], b=parsed_toml['simulation_parameters']['b'], scalar=parsed_toml['simulation_parameters']['weights_scalar'] / num_patterns)
w_ie = weights_ie(exc_n, parsed_toml['simulation_parameters']['inh_weights_scalar'], patterns, num_patterns)

combinations = list(itertools.product(*[i for i in parsed_toml['variables'].values()]))

all_states = [dict(zip(list(parsed_toml['variables'].keys()), combination)) for combination in combinations]

print(json.dumps(parsed_toml, indent=4))

e1 = 1
i1 = 0
c1 = 2

simulation_output = {}

for current_state in tqdm(all_states):
    trials = parsed_toml['simulation_parameters']['trials']
    pattern_indexes = list(
            itertools.chain(*[[i] * (trials // num_patterns) for i in range(num_patterns)])
        ) + \
        [int(i) for i in np.random.choice(range(num_patterns), trials % num_patterns)]
    # assert len(pattern_indexes) == trials
    
    for trial in range(trials):
        # pattern1 = np.random.choice(range(num_patterns), replace=False)
        pattern1 = pattern_indexes[trial]

        distortion = current_state['distortion']
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

        network = ln.DopaIzhikevichNetwork.generate_network([exc_lattice, inh_lattice], [spike_train_lattice])
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
        network.connect(2, 1, lambda x, y: x == y, lambda x, y: current_state['spike_train_to_exc'])
        network.set_dt(parsed_toml['simulation_parameters']['dt'])
        network.parallel = True

        network.apply_spike_train_lattice_given_position(
            c1, 
            get_spike_train_setup_function(
                patterns,
                pattern1, 
                current_state['distortion'],
                current_state['cue_firing_rate'],
                exc_n,
                parsed_toml['simulation_parameters']['distortion_on_only'],
            )
        )

        network.electrical_synapse = False
        network.chemical_synapse = True

        for _ in range(parsed_toml['simulation_parameters']['iterations']):
            network.run_lattices(1)

        hist = network.get_lattice(e1).history
        data = [i.flatten() for i in np.array(hist)]
        peaks = [find_peaks_above_threshold([j[i] for j in data], 20) for i in range(len(data[0]))]
        firing_rate_data = [int(len(i)) for i in peaks]

        current_state['trial'] = trial
        current_state['pattern'] = pattern1

        current_value = {}

        current_value['firing_rates'] = firing_rate_data

        signal = [float(i.mean()) for i in data]
        if parsed_toml['simulation_parameters']['voltages_on']:
            current_value['voltages'] = signal
        if parsed_toml['simulation_parameters']['measure_snr']:
            current_value['first_snr'] = float(
                signal_to_noise(signal[:parsed_toml['simulation_parameters']['iterations']])
            )

        key = generate_key(parsed_toml, current_state)

        simulation_output[key] = current_value

simulation_output['patterns'] = [[int(j) for j in i] for i in patterns]

with open(parsed_toml['simulation_parameters']['filename'], 'w') as file:
    json.dump(simulation_output, file, indent=4)

print("\033[92mFinished simulation\033[0m")
