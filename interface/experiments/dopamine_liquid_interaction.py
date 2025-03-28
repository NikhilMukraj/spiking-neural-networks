# regular disturbing spike train group to liquid
# second spike train group to liquid or inh
# option to turn on d1 or d2
# option to connect second spike train group to either liquid or inh
# vary parameters to determine effect of stability based on neuromodulation

import toml
import json
import sys
import itertools
import numpy as np
from tqdm import tqdm
from pipeline_setup import parse_toml, generate_key_helper, generate_setup_neuron, signal_to_noise, find_peaks_above_threshold
from lsm_setup import generate_liquid_weights, generate_start_firing, stop_firing, determine_return_to_baseline
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
    if 'settling_period' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['settling_period'] = 1000
    if 'tolerance' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['tolerance'] = 2

    if 'peaks_on' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['peaks_on'] = False

    if 'trials' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['trials'] = 10

    if 'skew' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['skew'] = 1

    if 'exc_n' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['exc_n'] = 7
    if 'inh_n' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['inh_n'] = 3

    if 'd1' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['d1'] = False
    if 'd2' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['d2'] = False

    if 'd_acts_on_inh' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['d_acts_on_inh'] = False

    if 'dt' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['dt'] = 1
    if 'c_m' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['c_m'] = 100

    if 'cue_firing_rate' not in parsed['variables']:
        parsed['variables']['cue_firing_rate'] = [0.01]
    if 'dopamine_firing_rate' not in parsed['variables']:
        parsed['variables']['dopamine_firing_rate'] = [0.01]

    if 'connectivity' not in parsed['variables']:
        parsed['variables']['connectivity'] = [0.25]
    if 'inh_connectivity' not in parsed['variables']:
        parsed['variables']['inh_connectivity'] = [0.25]
    if 'exc_to_inh_connectivity' not in parsed['variables']:
        parsed['variables']['exc_to_inh_connectivity'] = [0.15]
    if 'inh_to_exc_connectivity' not in parsed['variables']:
        parsed['variables']['inh_to_exc_connectivity'] = [0.15]
    if 'spike_train_connectivity' not in parsed['variables']:
        parsed['variables']['spike_train_connectivity'] = [0.5]
    
    if 'internal_scalar' not in parsed['variables']:
        parsed['variables']['internal_scalar'] = [0.125]
    if 'spike_train_to_exc' not in parsed['variables']:
        parsed['variables']['spike_train_to_exc'] = [3]
    if 'exc_to_inh_weight' not in parsed['variables']:
        parsed['variables']['exc_to_inh_weight'] = [0.0125]
    if 'inh_to_exc_weight' not in parsed['variables']:
        parsed['variables']['inh_to_exc_weight'] = [0.0125]
    if 'inh_internal_scalar' not in parsed['variables']:
        parsed['variables']['inh_internal_scalar'] = [2]

    if 'nmda_g' not in parsed['variables']:
        parsed['variables']['nmda_g'] = [0.6]
    if 'ampa_g' not in parsed['variables']:
        parsed['variables']['ampa_g'] = [1]
    if 'gabaa_g' not in parsed['variables']:
        parsed['variables']['gabaa_g'] = [1.2]

    if 's_d1' not in parsed['variables']:
        parsed['variables']['s_d1'] = [1]
    if 's_d2' not in parsed['variables']:
        parsed['variables']['s_d2'] = [0.025]

    if 'glutamate_clearance' not in parsed['variables']:
        parsed['variables']['glutamate_clearance'] = [0.001]
    if 'gabaa_clearance' not in parsed['variables']:
        parsed['variables']['gabaa_clearance'] = [0.001]
    if 'dopamine_clearance' not in parsed['variables']:
        parsed['variables']['dopamine_clearance'] = [0.001]

def generate_key(parsed, current_state):
    key = []

    key.append(f'trial: {current_state["trial"]}')

    fields = [
        'cue_firing_rate', 'dopamine_firing_rate',
        'connectivity', 'spike_train_connectivity', 'inh_connectivity', 'exc_to_inh_connectivity', 'inh_to_exc_connectivity',
        'spike_train_to_exc', 'internal_scalar', 'inh_internal_scalar', 'exc_to_inh_weight', 'inh_to_exc_weight',
        'nmda_g', 'ampa_g', 'gabaa_g', 's_d1', 's_d2',
        'glutamate_clearance', 'gabaa_clearance', 'dopamine_clearance',
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
inh_num = inh_n * inh_n

setup_neuron = generate_setup_neuron(
    parsed_toml['simulation_parameters']['c_m'], 
    parsed_toml['simulation_parameters']['skew'],
)

combinations = list(itertools.product(*[i for i in parsed_toml['variables'].values()]))

all_states = [dict(zip(list(parsed_toml['variables'].keys()), combination)) for combination in combinations]

print(json.dumps(parsed_toml, indent=4))

np.seterr(divide='ignore', invalid='ignore')

e1 = 0
i1 = 1
c1 = 2
c2 = 3

simulation_output = {}

for current_state in tqdm(all_states):
    for trial in range(parsed_toml['simulation_parameters']['trials']):
        w = generate_liquid_weights(
            num, connectivity=current_state['connectivity'], scalar=current_state['internal_scalar']
        )

        if not parsed_toml['simulation_parameters']['exc_only']:
            w_inh = generate_liquid_weights(
                inh_num, connectivity=current_state['inh_connectivity'], scalar=current_state['inh_internal_scalar']
            )

        start_firing = generate_start_firing(current_state['cue_firing_rate'])
        dopamine_firing = generate_start_firing(current_state['dopamine_firing_rate'])

        glu_neuro = ln.ApproximateNeurotransmitter(clearance_constant=current_state['glutamate_clearance'])
        exc_neurotransmitters = ln.DopaGluGABAApproximateNeurotransmitters()
        exc_neurotransmitters.set_neurotransmitter(ln.DopaGluGABANeurotransmitterType.Glutamate, glu_neuro)

        gaba_neuro = ln.ApproximateNeurotransmitter(clearance_constant=current_state['gabaa_clearance'])
        inh_neurotransmitters = ln.DopaGluGABAApproximateNeurotransmitters()
        inh_neurotransmitters.set_neurotransmitter(ln.DopaGluGABANeurotransmitterType.GABA, gaba_neuro)

        dopa_neuro = ln.ApproximateNeurotransmitter(clearance_constant=current_state['dopamine_clearance'])
        dopamine_neurotransmitters = ln.DopaGluGABAApproximateNeurotransmitters()
        dopamine_neurotransmitters.set_neurotransmitter(ln.DopaGluGABANeurotransmitterType.Dopamine, dopa_neuro)

        glu = ln.GlutamateReceptor()
        
        glu.ampa_g = current_state['nmda_g']
        glu.nmda_g = current_state['ampa_g']

        gaba = ln.GABAReceptor()
        gaba.g = current_state['gabaa_g']

        dopamine_rs = ln.DopamineReceptor()

        dopamine_rs.d1_enabled = parsed_toml['simulation_parameters']['d1']
        dopamine_rs.d2_enabled = parsed_toml['simulation_parameters']['d2']

        dopamine_rs.s_d1 = current_state['s_d1']
        dopamine_rs.s_d2 = current_state['s_d2']

        receptors = ln.DopaGluGABAReceptors()

        receptors.set_receptor(ln.DopaGluGABANeurotransmitterType.Glutamate, glu)
        receptors.set_receptor(ln.DopaGluGABANeurotransmitterType.GABA, gaba)
        receptors.set_receptor(ln.DopaGluGABANeurotransmitterType.Dopamine, dopamine_rs)

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

        dopamine_poisson_neuron = ln.DopaPoissonNeuron()
        dopamine_poisson_neuron.set_neurotransmitters(dopamine_neurotransmitters)

        dopamine_spike_train_lattice = ln.DopaPoissonLattice(c2)
        dopamine_spike_train_lattice.populate(dopamine_poisson_neuron, exc_n, exc_n)

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
                [exc_lattice, inh_lattice], [spike_train_lattice, dopamine_spike_train_lattice],
            )
        else:
            network = ln.DopaIzhikevichNetwork.generate_network(
                [exc_lattice], [spike_train_lattice, dopamine_spike_train_lattice],
            )

        network.set_dt(parsed_toml['simulation_parameters']['dt'])
        network.parallel = True

        network.electrical_synapse = False
        network.chemical_synapse = True

        network.apply_spike_train_lattice(
            c2,
            dopamine_firing
        )

        if not parsed_toml['simulation_parameters']['exc_only']:
            network.connect(
                i1, 
                e1, 
                lambda x, y: np.random.uniform(0, 1) < current_state['inh_to_exc_connectivity'], 
                lambda x, y: current_state['inh_to_exc_weight'],
            )
            network.connect(
                i1, 
                e1, 
                lambda x, y: np.random.uniform(0, 1) < current_state['exc_to_inh_connectivity'],
                lambda x, y: current_state['exc_to_inh_weight'],
            )

        network.connect(
            c1, 
            e1, 
            lambda x, y: np.random.uniform(0, 1) < current_state['spike_train_connectivity'], 
            lambda x, y: current_state['spike_train_to_exc']
        )

        if parsed_toml['simulation_parameters']['d_acts_on_inh']:
            lattice_to_connect_to = i1
        else:
            lattice_to_connect_to = e1

        network.connect(
            c2, 
            e1, 
            lambda x, y: np.random.uniform(0, 1) < current_state['spike_train_connectivity'], 
            lambda x, y: current_state['spike_train_to_exc']
        )

        network.electrical_synapse = False
        network.chemical_synapse = True

        network.apply_spike_train_lattice(
            c1,
            stop_firing
        )
        network.run_lattices(parsed_toml['simulation_parameters']['off_phase'])

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

        hist = network.get_lattice(0).history
        voltages = [float(np.array(i).mean()) for i in hist]

        return_to_baseline = determine_return_to_baseline(
            voltages,
            parsed_toml['simulation_parameters']['settling_period'],
            parsed_toml['simulation_parameters']['on_phase'],
            parsed_toml['simulation_parameters']['off_phase'],
            parsed_toml['simulation_parameters']['tolerance'],
        )

        current_value = {}

        if parsed_toml['simulation_parameters']['measure_snr']:
            current_value['first_snr'] = float(
                signal_to_noise(voltages[
                        parsed_toml['simulation_parameters']['settling_period']:parsed_toml['simulation_parameters']['off_phase']
                    ]
                )
            )
            current_value['second_snr'] = float(
                signal_to_noise(voltages[
                        parsed_toml['simulation_parameters']['on_phase'] + parsed_toml['simulation_parameters']['off_phase']:
                    ]
                )
            )
            current_value['during_disturbance'] = float(
                signal_to_noise(voltages[
                        parsed_toml['simulation_parameters']['on_phase']:
                        parsed_toml['simulation_parameters']['on_phase'] + parsed_toml['simulation_parameters']['off_phase']:
                    ]
                )
            )

        current_value['return_to_baseline'] = return_to_baseline
        current_value['voltages'] = voltages

        if parsed_toml['simulation_parameters']['peaks_on']:
            hist = network.get_lattice(0).history
            data = [i.flatten() for i in np.array(hist)]
            peaks = [find_peaks_above_threshold([j[i] for j in data], 20) for i in range(len(data[0]))]

            current_value['peaks'] = [[int(item) for item in sublist] for sublist in peaks]

        current_state['trial'] = trial

        key = generate_key(parsed_toml, current_state)

        simulation_output[key] = current_value

with open(parsed_toml['simulation_parameters']['filename'], 'w') as file:
    json.dump(simulation_output, file, indent=4)

print("\033[92mFinished simulation\033[0m")
