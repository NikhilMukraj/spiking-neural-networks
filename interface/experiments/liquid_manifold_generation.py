# generate liquid
# input random digit from mnist
# record trials
# generate umap based on neural activity
# determine which neurons contribute to which digits by
# inputting differing firing rates into reducer to see where they land

import json
import sys
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from pipeline_setup import parse_toml, generate_setup_neuron, find_peaks_above_threshold
from lsm_setup import generate_liquid_weights, generate_start_firing, stop_firing
import lixirnet as ln


def fill_defaults(parsed):
    if 'simulation_parameters' not in parsed:
        raise ValueError('Requires `simulation_parameters` table')

    if 'filename' not in parsed['simulation_parameters']:
        raise ValueError('Requires `filename` field in `simulation_parameters`')
    
    if 'variables' not in parsed:
        raise ValueError('Requires `variables` table')

    if 'exc_only' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['exc_only'] = True

    if 'on_phase' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['on_phase'] = 1000
    if 'off_phase' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['off_phase'] = 5000
   
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

    if 'percentage_sample' not in parsed['variables']:
        parsed['variables']['percentage_sample'] = 0.1
    if 'spacing_term' not in parsed['variables']:
        parsed['variables']['spacing_term'] = 3

    if 'cue_firing_rate' not in parsed['variables']:
        parsed['variables']['cue_firing_rate'] = 0.01

    if 'connectivity' not in parsed['variables']:
        parsed['variables']['connectivity'] = 0.25
    if 'inh_connectivity' not in parsed['variables']:
        parsed['variables']['inh_connectivity'] = 0.25
    if 'exc_to_inh_connectivity' not in parsed['variables']:
        parsed['variables']['exc_to_inh_connectivity'] = 0.15
    if 'inh_to_exc_connectivity' not in parsed['variables']:
        parsed['variables']['inh_to_exc_connectivity'] = 0.15
    if 'spike_train_connectivity' not in parsed['variables']:
        parsed['variables']['spike_train_connectivity'] = 0.5
    
    if 'internal_scalar' not in parsed['variables']:
        parsed['variables']['internal_scalar'] = 0.5
    if 'spike_train_to_exc' not in parsed['variables']:
        parsed['variables']['spike_train_to_exc'] = 3
    if 'exc_to_inh_weight' not in parsed['variables']:
        parsed['variables']['exc_to_inh_weight'] = 0.0125
    if 'inh_to_exc_weight' not in parsed['variables']:
        parsed['variables']['inh_to_exc_weight'] = 0.0125
    if 'inh_internal_scalar' not in parsed['variables']:
        parsed['variables']['inh_internal_scalar'] = 2

    if 'nmda_g' not in parsed['variables']:
        parsed['variables']['nmda_g'] = 0.6
    if 'ampa_g' not in parsed['variables']:
        parsed['variables']['ampa_g'] = 1
    if 'gabaa_g' not in parsed['variables']:
        parsed['variables']['gabaa_g'] = 1.2

    if 'glutamate_clearance' not in parsed['variables']:
        parsed['variables']['glutamate_clearance'] = 0.001
    if 'gabaa_clearance' not in parsed['variables']:
        parsed['variables']['gabaa_clearance'] = 0.001

with open(sys.argv[1], 'r') as f:
    parsed_toml = parse_toml(f)

fill_defaults(parsed_toml)

exc_n = parsed_toml['simulation_parameters']['exc_n']
num = exc_n * exc_n

inh_n = parsed_toml['simulation_parameters']['inh_n']

setup_neuron = generate_setup_neuron(
    parsed_toml['simulation_parameters']['c_m'], 
    0.1,
)

digits = load_digits()

percentage_sample = parsed_toml['variables']['percentage_sample']
subset_size = int(percentage_sample * len(digits.data))

data, _, target, _ = train_test_split(
    digits.data, digits.target, train_size=subset_size, stratify=digits.target
)

spacing_term = parsed_toml['variables']['spacing_term']
digits_size = 8

# weights for cue to liquid
cue_to_liquid = np.array([
    [i % spacing_term == 0 for i in range(digits_size * spacing_term)] 
    for _ in range(digits_size * spacing_term)
])

w = generate_liquid_weights(
    num, connectivity=parsed_toml['variables']['connectivity'], scalar=parsed_toml['variables']['internal_scalar']
)

if not parsed_toml['simulation_parameters']['exc_only']:
    w_inh = generate_liquid_weights(
        inh_num, connectivity=parsed_toml['variables']['inh_connectivity'], scalar=parsed_toml['variables']['inh_internal_scalar']
    )

print(json.dumps(parsed_toml, indent=4))

e1 = 0
i1 = 1
c1 = 2

simulation_output = {}

for current_digit, current_class in zip(data, target):

    start_firing = generate_start_firing(parsed_toml['variables']['cue_firing_rate'])

    glu_neuro = ln.ApproximateNeurotransmitter(clearance_constant=parsed_toml['variables']['glutamate_clearance'])
    exc_neurotransmitters = ln.DopaGluGABAApproximateNeurotransmitters()
    exc_neurotransmitters.set_neurotransmitter(ln.DopaGluGABANeurotransmitterType.Glutamate, glu_neuro)

    gaba_neuro = ln.ApproximateNeurotransmitter(clearance_constant=parsed_toml['variables']['gabaa_clearance'])
    inh_neurotransmitters = ln.DopaGluGABAApproximateNeurotransmitters()
    inh_neurotransmitters.set_neurotransmitter(ln.DopaGluGABANeurotransmitterType.GABA, gaba_neuro)

    glu = ln.GlutamateReceptor()
    
    glu.ampa_g = parsed_toml['variables']['nmda_g']
    glu.nmda_g = parsed_toml['variables']['ampa_g']

    gaba = ln.GABAReceptor()
    gaba.g = parsed_toml['variables']['gabaa_g']

    receptors = ln.DopaGluGABAReceptors()

    receptors.set_receptor(ln.DopaGluGABANeurotransmitterType.Glutamate, glu)
    receptors.set_receptor(ln.DopaGluGABANeurotransmitterType.GABA, gaba)

    exc_neuron = ln.DopaIzhikevichNeuron()
    exc_neuron.set_neurotransmitters(exc_neurotransmitters)
    exc_neuron.set_receptors(receptors)

    inh_neuron = ln.DopaIzhikevichNeuron()
    inh_neuron.set_neurotransmitters(inh_neurotransmitters)
    inh_neuron.set_receptors(receptors)

    poisson_neuron = ln.DopaPoissonNeuron()
    poisson_neuron.set_neurotransmitters(exc_neurotransmitters)

    exc_lattice = ln.DopaIzhikevichLattice(e1)
    exc_lattice.populate(exc_neuron, exc_n, exc_n)
    exc_lattice.apply(setup_neuron)
    position_to_index = exc_lattice.position_to_index
    exc_lattice.connect(
        lambda x, y: bool(float(w[position_to_index[x]][position_to_index[y]]) != 0), 
        lambda x, y: float(w[position_to_index[x]][position_to_index[y]]),
    )
    exc_lattice.update_grid_history = True

    spike_train_lattice = ln.DopaPoissonLattice(c1)
    spike_train_lattice.populate(poisson_neuron, exc_n, exc_n)

    if not parsed_toml['simulation_parameters']['exc_only']:
        inh_lattice = ln.DopaIzhikevichLattice(i1)
        inh_lattice.populate(inh_neuron, inh_n, inh_n)
        inh_lattice.apply(setup_neuron)
        position_to_index = inh_lattice.position_to_index
        inh_lattice.connect(
            lambda x, y: bool(float(w_inh[position_to_index[x]][position_to_index[y]]) != 0), 
            lambda x, y: float(w_inh[position_to_index[x]][position_to_index[y]]),
        )
        # inh_lattice.update_grid_history = True

        network = ln.DopaIzhikevichNetwork.generate_network(
            [exc_lattice, inh_lattice], [spike_train_lattice],
        )
    else:
        network = ln.DopaIzhikevichNetwork.generate_network(
            [exc_lattice], [spike_train_lattice],
        )

    network.set_dt(parsed_toml['simulation_parameters']['dt'])
    network.parallel = True

    if not parsed_toml['simulation_parameters']['exc_only']:
        network.connect(
            i1, 
            e1, 
            lambda x, y: np.random.uniform(0, 1) < parsed_toml['variables']['inh_to_exc_connectivity'], 
            lambda x, y: parsed_toml['variables']['inh_to_exc_weight'],
        )
        network.connect(
            e1, 
            i1, 
            lambda x, y: np.random.uniform(0, 1) < parsed_toml['variables']['exc_to_inh_connectivity'],
            lambda x, y: parsed_toml['variables']['exc_to_inh_weight'],
        )

    network.electrical_synapse = False
    network.chemical_synapse = True

    network.run_lattices(parsed_toml['simulation_parameters']['off_phase'])

    network.connect(
        c1, 
        e1, 
        lambda x, y: bool(cue_to_liquid[x[0]][x[1]]), 
        lambda x, y: parsed_toml['variables']['spike_train_to_exc']
    )

    network.apply_spike_train_lattice(
        c1,
        start_firing
    )

    network.run_lattices(parsed_toml['simulation_parameters']['on_phase'])

    network.apply_spike_train_lattice(
        c1,
        stop_firing
    )

    network.run_lattices(parsed_toml['simulation_parameters']['off_phase'])

    network.apply_spike_train_lattice(
        c1,
        start_firing
    )

    hist = network.get_lattice(e1).history
    data = [i.flatten() for i in np.array(hist)]
    peaks = [find_peaks_above_threshold([j[i] for j in data], 20) for i in range(len(data[0]))]

    current_value = {}

    current_value['firing_rates'] = [int(len(i)) for i in peaks]
    current_value['peaks'] = peaks
    signal = [float(i.mean()) for i in data]
    current_value['voltages'] = signal

    simulation_output[(str(current_digit), str(current_class))] = current_value

with open(parsed_toml['simulation_parameters']['filename'], 'w') as file:
    json.dump(simulation_output, file, indent=4)

print("\033[92mFinished simulation\033[0m")

