# generate liquid with random weights, one exc group and one inh group
# vary glu, gaba conductance and clearance
# test how long disturbances last in the liquid

# off phase
# on phase
# time to wait for settling
# iterate over all timesteps after disturbance is removed
# determine the first timestep where average voltage becomes in range with baseline voltage
# baseline voltage should wait for the liquid to settle first

import toml
import json
import sys
import itertools
import numpy as np
from tqdm import tqdm
from pipeline_setup import generate_setup_neuron, signal_to_noise
from lsm_setup import generate_liquid_weights
import lixirnet as ln


def fill_defaults(parsed):
    if 'simulation_parameters' not in parsed:
        raise ValueError('Requires `simulation_parameters` table')

    if 'filename' not in parsed['simulation_parameters']:
        raise ValueError('Requires `filename` field in `simulation_parameters`')
    
    if 'variables' not in parsed:
        raise ValueError('Requires `variables` table')

    if 'on_phase' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['on_phase'] = 1000
    if 'off_phase' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['off_phase'] = 5000
    if 'settling_period' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['settling_period'] = 1000
    if 'tolerance' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['tolerance'] = 2

    if 'trials' not in parsed['simulation_parameters']:
        parsed['simulation_parameters']['trials'] = 10

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

    if 'cue_firing_rate' not in parsed['variables']:
        parsed['variables']['cue_firing_rate'] = 0.01

    if 'connectivity' not in parsed['variables']:
        parsed['variables']['connectivity'] = 0.25
    if 'spike_train_connectivity' not in parsed['variables']:
        parsed['variables']['spike_train_connectivity'] = 0.5
    
    if 'internal_scalar' not in parsed['variables']:
        parsed['variables']['weights_scalar'] = 0.5
    if 'spike_train_to_exc' not in parsed['variables']:
        parsed['variables']['spike_train_to_exc'] = 3

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
    if 'dopamine_clearance' not in parsed['variables']:
        parsed['variables']['dopamine_clearance'] = [0.001]

def generate_key(parsed, current_state):
    key = []

    key.append(f'trial: {current_state["trial"]}')

    fields = [
        'cue_firing_rate', 'connectivity', 'spike_train_connectivity'
        'spike_train_to_exc', 'internal_scalar'
        'nmda_g', 'ampa_g', 'gabaa_g',
        'glutamate_clearance', 'gabaa_clearance', 
    ]
    
    for field in fields:
        generate_key_helper(current_state, key, parsed, field)

    return ', '.join(key)

def start_firing(neuron):
    neuron.chance_of_firing = 0.01

    return neuron

def stop_firing(neuron):
    neuron.chance_of_firing = 0

    return neuron

def determine_return_to_baseline(voltages, settling_period, on_phase, off_phase, tolerance):
    baseline = np.array(voltages[1000:off_phase]).mean()

    for i in range(off_phase):
        current_voltage_average = np.array(voltages[off_phase + on_phase + i:]).mean()
        if abs(baseline - current_voltage_average) < tolerance:
            return i
    
    return off_phase

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

combinations = list(itertools.product(*[i for i in parsed_toml['variables'].values()]))

all_states = [dict(zip(list(parsed_toml['variables'].keys()), combination)) for combination in combinations]

print(json.dumps(parsed_toml, indent=4))

np.seterr(divide='ignore', invalid='ignore')

simulation_output = {}

for current_state in tqdm(all_states):
    for trial in parsed_toml['simulation_parameters']['trials']:
        w = generate_liquid_weights(
            num, connectivity=current_state['connectivity'], scalar=current_state['internal_scalar']
        )

        glu_neuro = ln.ApproximateNeurotransmitter(clearance_constant=current_state['glu_clearance'])
        exc_neurotransmitters = ln.DopaGluGABAApproximateNeurotransmitters()
        exc_neurotransmitters.set_neurotransmitter(ln.DopaGluGABANeurotransmitterType.Glutamate, glu_neuro)

        glu = ln.GlutamateReceptor()
        
        glu.ampa_g = current_state['nmda_g']
        glu.nmda_g = current_state['ampa_g']

        receptors = ln.DopaGluGABAReceptors()

        receptors.set_receptor(ln.DopaGluGABANeurotransmitterType.Glutamate, glu)

        exc_neuron = ln.DopaIzhikevichNeuron()
        exc_neuron.set_neurotransmitters(exc_neurotransmitters)
        exc_neuron.set_receptors(receptors)

        poisson_neuron = ln.DopaPoissonNeuron()
        poisson_neuron.set_neurotransmitters(exc_neurotransmitters)

        exc_lattice = ln.DopaIzhikevichLattice(0)
        exc_lattice.populate(exc_neuron, exc_n, exc_n)
        exc_lattice.apply(setup_neuron)
        position_to_index = exc_lattice.position_to_index
        exc_lattice.connect(
            lambda x, y: bool(float(w[position_to_index[x]][position_to_index[y]]) != 0), 
            lambda x, y: float(w[position_to_index[x]][position_to_index[y]]),
        )
        exc_lattice.update_grid_history = True

        spike_train_lattice = ln.DopaPoissonLattice(1)
        spike_train_lattice.populate(poisson_neuron, exc_n, exc_n)

        network = ln.DopaIzhikevichNetwork.generate_network(
            [exc_lattice], [spike_train_lattice],
        )
        network.set_dt(parsed_toml['simulation_parameters']['dt'])
        network.parallel = True

        network.connect(
            1, 
            0, 
            lambda x, y: np.random.uniform(0, 1) < current_state['spike_train_connectivity'], 
            lambda x, y: current_state['spike_train_to_exc']
        )

        network.electrical_synapse = False
        network.chemical_synapse = True

        network.apply_spike_train_lattice(
            1,
            stop_firing
        )
        network.run_lattices(off_phase)

        network.apply_spike_train_lattice(
            1,
            start_firing
        )
        network.run_lattices(on_phase)

        network.apply_spike_train_lattice(
            1,
            stop_firing
        )
        network.run_lattices(off_phase)

        hist = network.get_lattice(0).history
        voltages = [float(np.array(i).mean()) for i in hist]

        return_to_baseline = determine_return_to_baseline(
            voltages,
            parsed_toml['simulation_parameters']['settling_period'],
            parsed_toml['simulation_parameters']['on_phase'],
            parsed_toml['simulation_parameters']['off_phase'],
            parsed_toml['simulation_parameters']['tolerance'],
        )

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

        key = generate_key(parsed_toml, current_state)

        simulation_output[key] = current_value

with open(parsed_toml['simulation_parameters']['filename'], 'w') as file:
    json.dump(simulation_output, file, indent=4)

print("\033[92mFinished simulation\033[0m")
