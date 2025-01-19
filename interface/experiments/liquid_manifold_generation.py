# generate liquid
# input random digit from mnist
# record trials
# generate umap based on neural activity
# determine which neurons contribute to which digits by
# inputting differing firing rates into reducer to see where they land

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


digits = load_digits()

percentage_sample = 0.1
subset_size = int(percentage_sample * len(digits.data))

data, _, target, _ = train_test_split(
    digits.data, digits.target, train_size=subset_size, stratify=digits.target
)

spacing_term = 3
digits_size = 8

# weights for cue to liquid
cue_to_liquid = np.array([
    [i % spacing_term == 0 for i in range(digits_size * spacing_term)] 
    for _ in range(digits_size * spacing_term)
])

# map cue to liquid with above weight matrix

# for current_state in tqdm(all_states):
#     for trial in range(parsed_toml['simulation_parameters']['trials']):
#         w = generate_liquid_weights(
#             num, connectivity=current_state['connectivity'], scalar=current_state['internal_scalar']
#         )

#         if not parsed_toml['simulation_parameters']['exc_only']:
#             w_inh = generate_liquid_weights(
#                 inh_num, connectivity=current_state['inh_connectivity'], scalar=current_state['inh_internal_scalar']
#             )

#         start_firing = generate_start_firing(current_state['cue_firing_rate'])

#         glu_neuro = ln.ApproximateNeurotransmitter(clearance_constant=current_state['glu_clearance'])
#         exc_neurotransmitters = ln.DopaGluGABAApproximateNeurotransmitters()
#         exc_neurotransmitters.set_neurotransmitter(ln.DopaGluGABANeurotransmitterType.Glutamate, glu_neuro)

#         gaba_neuro = ln.ApproximateNeurotransmitter(clearance_constant=current_state['gabaa_clearance'])
#         inh_neurotransmitters = ln.DopaGluGABAApproximateNeurotransmitters()
#         inh_neurotransmitters.set_neurotransmitter(ln.DopaGluGABANeurotransmitterType.GABA, gaba_neuro)

#         glu = ln.GlutamateReceptor()
        
#         glu.ampa_g = current_state['nmda_g']
#         glu.nmda_g = current_state['ampa_g']

#         gaba = ln.GABAReceptor()
#         gaba.g = current_state['gabaa_g']

#         receptors = ln.DopaGluGABAReceptors()

#         receptors.set_receptor(ln.DopaGluGABANeurotransmitterType.Glutamate, glu)
#         receptors.set_receptor(ln.DopaGluGABANeurotransmitterType.GABA, gaba)

#         exc_neuron = ln.DopaIzhikevichNeuron()
#         exc_neuron.set_neurotransmitters(exc_neurotransmitters)
#         exc_neuron.set_receptors(receptors)

#         inh_neuron = ln.DopaIzhikevichNeuron()
#         inh_neuron.set_neurotransmitters(inh_neurotransmitters)
#         inh_neuron.set_receptors(receptors)

#         poisson_neuron = ln.DopaPoissonNeuron()
#         poisson_neuron.set_neurotransmitters(exc_neurotransmitters)

#         exc_lattice = ln.DopaIzhikevichLattice(0)
#         exc_lattice.populate(exc_neuron, exc_n, exc_n)
#         exc_lattice.apply(setup_neuron)
#         position_to_index = exc_lattice.position_to_index
#         exc_lattice.connect(
#             lambda x, y: bool(float(w[position_to_index[x]][position_to_index[y]]) != 0), 
#             lambda x, y: float(w[position_to_index[x]][position_to_index[y]]),
#         )
#         exc_lattice.update_grid_history = True

#         spike_train_lattice = ln.DopaPoissonLattice(1)
#         spike_train_lattice.populate(poisson_neuron, exc_n, exc_n)

#         if not parsed_toml['simulation_parameters']['exc_only']:
#             inh_lattice = ln.DopaIzhikevichLattice(2)
#             inh_lattice.populate(inh_neuron, inh_n, inh_n)
#             inh_lattice.apply(setup_neuron)
#             position_to_index = inh_lattice.position_to_index
#             inh_lattice.connect(
#                 lambda x, y: bool(float(w_inh[position_to_index[x]][position_to_index[y]]) != 0), 
#                 lambda x, y: float(w_inh[position_to_index[x]][position_to_index[y]]),
#             )
#             # inh_lattice.update_grid_history = True

#             network = ln.DopaIzhikevichNetwork.generate_network(
#                 [exc_lattice, inh_lattice], [spike_train_lattice],
#             )
#         else:
#             network = ln.DopaIzhikevichNetwork.generate_network(
#                 [exc_lattice], [spike_train_lattice],
#             )

#         network.set_dt(parsed_toml['simulation_parameters']['dt'])
#         network.parallel = True

#         if not parsed_toml['simulation_parameters']['exc_only']:
#             network.connect(
#                 2, 
#                 0, 
#                 lambda x, y: np.random.uniform(0, 1) < current_state['inh_to_exc_connectivity'], 
#                 lambda x, y: current_state['inh_to_exc_weight'],
#             )
#             network.connect(
#                 0, 
#                 2, 
#                 lambda x, y: np.random.uniform(0, 1) < current_state['exc_to_inh_connectivity'],
#                 lambda x, y: current_state['exc_to_inh_weight'],
#             )

#         network.connect(
#             1, 
#             0, 
#             lambda x, y: np.random.uniform(0, 1) < current_state['spike_train_connectivity'], 
#             lambda x, y: current_state['spike_train_to_exc']
#         )

#         network.electrical_synapse = False
#         network.chemical_synapse = True
